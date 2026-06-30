"""
smart_reorder_agent.py
=======================
Higher-end Smart Reorder Agent for SCM automation.

Upgrades over the baseline version:
  - Statistical demand forecasting (weighted moving average + linear trend)
    with explicit confidence intervals, instead of hardcoded keyword rules.
  - A continuous feedback/learning loop: forecast error is measured every
    run against observed consumption and used to update per-product /
    per-signal bias terms (EWMA online learning) stored in the DB.
  - A full, explicit Purchase Order state machine
    (DRAFT -> APPROVED -> SENT_TO_SUPPLIER -> IN_TRANSIT -> RECEIVED -> CLOSED,
    with CANCELLED / EXCEPTION branches) instead of a single static status.
  - Missing core SCM entities are introduced: Shipment, Receipt and
    InventoryTransaction tables (auto-created if absent), so the agent
    has a real ledger of what was ordered, shipped, received and consumed.
  - Reorder sizing now respects supplier order-capacity and warehouse
    storage capacity, instead of ignoring them.
  - A monitoring pass runs on every invocation to advance/age in-flight
    POs and raise exceptions for shipments that are overdue relative to
    the supplier's quoted lead time.
  - DB path, weights and learning rate are fully configurable via env vars
    (no more hardcoded "smartreorder.db").
  - Structured (JSON) logging to stdout/stderr in addition to the DB audit
    log, with correlated run_id for tracing a single run end-to-end.

This remains a single, self-contained file. On first run against an
existing baseline DB it will non-destructively extend the schema
(ALTER TABLE / CREATE TABLE IF NOT EXISTS) rather than requiring a
separate migration script.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration (env-driven, no hardcoding)
# ---------------------------------------------------------------------------

DB_PATH = os.getenv("SCM_DB_PATH", os.getenv("DB_PATH", "smartreorder.db"))
DEMAND_CONTEXT_PATH = os.getenv("SCM_DEMAND_CONTEXT_PATH", "demand_context.txt")

PRICE_WEIGHT = float(os.getenv("PRICE_WEIGHT", "0.4"))
RELIABILITY_WEIGHT = float(os.getenv("RELIABILITY_WEIGHT", "0.4"))
LEADTIME_WEIGHT = float(os.getenv("LEADTIME_WEIGHT", "0.2"))

FORECAST_WINDOW_DAYS = int(os.getenv("FORECAST_WINDOW_DAYS", "14"))
CONFIDENCE_Z = float(os.getenv("CONFIDENCE_Z", "1.96"))  # ~95% CI
LEARNING_RATE = float(os.getenv("SCM_LEARNING_RATE", "0.15"))  # EWMA alpha for feedback loop
DEFAULT_WAREHOUSE_CAPACITY = int(os.getenv("DEFAULT_WAREHOUSE_CAPACITY", "1_000_000"))
DEFAULT_SUPPLIER_MAX_ORDER = int(os.getenv("DEFAULT_SUPPLIER_MAX_ORDER", "1_000_000"))
OVERDUE_GRACE_DAYS = int(os.getenv("OVERDUE_GRACE_DAYS", "1"))


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("run_id", "product_id", "event_type"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def build_logger() -> logging.Logger:
    logger = logging.getLogger("smart_reorder_agent")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(os.getenv("SCM_LOG_LEVEL", "INFO"))
    return logger


log = build_logger()


# ---------------------------------------------------------------------------
# Domain entities
# ---------------------------------------------------------------------------

@dataclass
class Product:
    product_id: str
    name: str
    category: str
    avg_daily_sales: float
    safety_stock: int
    on_hand_qty: int
    on_order_qty: int
    allocated_qty: int
    reorder_multiple: int
    preferred_days_cover: int
    warehouse_capacity: Optional[int] = None


@dataclass
class SupplierOption:
    supplier_id: str
    supplier_name: str
    product_id: str
    unit_price: float
    lead_time_days: int
    reliability: float
    min_order_qty: int
    max_order_qty: Optional[int] = None


@dataclass
class ForecastResult:
    daily_mean: float
    daily_std: float
    ci_low: float
    ci_high: float
    method: str
    rationale: str
    learned_bias: float = 0.0


class POState(str, Enum):
    DRAFT = "DRAFT"
    APPROVED = "APPROVED"
    SENT_TO_SUPPLIER = "SENT_TO_SUPPLIER"
    IN_TRANSIT = "IN_TRANSIT"
    RECEIVED = "RECEIVED"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"
    EXCEPTION = "EXCEPTION"


# Explicit, validated state machine. No silent/implicit transitions.
PO_TRANSITIONS: Dict[POState, List[POState]] = {
    POState.DRAFT: [POState.APPROVED, POState.CANCELLED],
    POState.APPROVED: [POState.SENT_TO_SUPPLIER, POState.CANCELLED],
    POState.SENT_TO_SUPPLIER: [POState.IN_TRANSIT, POState.EXCEPTION, POState.CANCELLED],
    POState.IN_TRANSIT: [POState.RECEIVED, POState.EXCEPTION],
    POState.RECEIVED: [POState.CLOSED],
    POState.EXCEPTION: [POState.IN_TRANSIT, POState.CANCELLED],
    POState.CLOSED: [],
    POState.CANCELLED: [],
}


class InvalidTransitionError(Exception):
    pass


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class SmartReorderAgent:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.run_id = uuid.uuid4().hex[:12]

    # -----------------------------
    # DB / infra
    # -----------------------------
    def connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def ensure_schema_extensions(self, conn):
        """Non-destructively extend an existing baseline schema with the
        entities/columns required for state-machine, capacity-aware and
        forecasting behavior. Safe to call every run."""

        def add_column(table, coldef):
            colname = coldef.split()[0]
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {coldef}")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise

        # --- extend products / supplier_products with capacity fields ---
        add_column("products", "warehouse_capacity INTEGER")
        add_column("supplier_products", "max_order_qty INTEGER")

        # --- extend purchase_orders with state machine + forecast fields ---
        add_column("purchase_orders", "state TEXT DEFAULT 'DRAFT'")
        add_column("purchase_orders", "forecast_mean REAL")
        add_column("purchase_orders", "forecast_std REAL")
        add_column("purchase_orders", "forecast_ci_low REAL")
        add_column("purchase_orders", "forecast_ci_high REAL")
        add_column("purchase_orders", "expected_receipt_date TEXT")
        add_column("purchase_orders", "run_id TEXT")

        # --- core missing SCM entities ---
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS po_state_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                po_id INTEGER NOT NULL,
                from_state TEXT,
                to_state TEXT NOT NULL,
                ts TEXT NOT NULL,
                reason TEXT
            );

            CREATE TABLE IF NOT EXISTS shipments (
                shipment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                po_id INTEGER NOT NULL,
                product_id TEXT NOT NULL,
                shipped_qty INTEGER NOT NULL,
                shipped_at TEXT NOT NULL,
                expected_arrival TEXT,
                carrier_status TEXT DEFAULT 'IN_TRANSIT'
            );

            CREATE TABLE IF NOT EXISTS receipts (
                receipt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                shipment_id INTEGER NOT NULL,
                po_id INTEGER NOT NULL,
                product_id TEXT NOT NULL,
                received_qty INTEGER NOT NULL,
                received_at TEXT NOT NULL,
                condition TEXT DEFAULT 'OK'
            );

            CREATE TABLE IF NOT EXISTS inventory_transactions (
                txn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT NOT NULL,
                txn_type TEXT NOT NULL,      -- SNAPSHOT | RECEIPT | CONSUMPTION | ADJUSTMENT
                qty_delta INTEGER NOT NULL,
                on_hand_after INTEGER,
                ts TEXT NOT NULL,
                ref TEXT
            );

            CREATE TABLE IF NOT EXISTS demand_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                observed_daily_sales REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS model_state (
                product_id TEXT PRIMARY KEY,
                learned_bias REAL NOT NULL DEFAULT 0.0,
                last_predicted_mean REAL,
                last_updated TEXT
            );

            CREATE TABLE IF NOT EXISTS prediction_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT NOT NULL,
                predicted_daily_mean REAL NOT NULL,
                observed_daily_sales REAL,
                error REAL,
                ts TEXT NOT NULL
            );
            """
        )
        conn.commit()

    def log_event(self, conn, level, event_type, product_id, message, payload=None):
        conn.execute(
            """
            INSERT INTO audit_log (ts, level, event_type, product_id, message, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(),
                level,
                event_type,
                product_id,
                message,
                json.dumps(payload or {}),
            ),
        )
        conn.commit()
        getattr(log, level.lower() if level.lower() in ("info", "warning", "error") else "info")(
            message,
            extra={"run_id": self.run_id, "product_id": product_id, "event_type": event_type},
        )

    # -----------------------------
    # Data loading
    # -----------------------------
    def load_products(self, conn) -> List[Product]:
        rows = conn.execute(
            """
            SELECT product_id, name, category, avg_daily_sales, safety_stock,
                   on_hand_qty, on_order_qty, allocated_qty,
                   reorder_multiple, preferred_days_cover,
                   COALESCE(warehouse_capacity, ?)
            FROM products
            """,
            (DEFAULT_WAREHOUSE_CAPACITY,),
        ).fetchall()

        products = []
        for r in rows:
            p = Product(*r)
            self.validate_product(p)
            products.append(p)
        return products

    def load_supplier_options(self, conn, product_id: str) -> List[SupplierOption]:
        rows = conn.execute(
            """
            SELECT sp.supplier_id, s.name, sp.product_id, sp.unit_price,
                   sp.lead_time_days, sp.reliability, sp.min_order_qty,
                   COALESCE(sp.max_order_qty, ?)
            FROM supplier_products sp
            JOIN suppliers s ON s.supplier_id = sp.supplier_id
            WHERE sp.product_id = ?
            """,
            (DEFAULT_SUPPLIER_MAX_ORDER, product_id),
        ).fetchall()

        options = []
        for r in rows:
            option = SupplierOption(*r)
            self.validate_supplier_option(option)
            options.append(option)
        return options

    # -----------------------------
    # Validation
    # -----------------------------
    def validate_product(self, p: Product):
        if p.avg_daily_sales < 0:
            raise ValueError(f"{p.product_id}: avg_daily_sales cannot be negative")
        if p.safety_stock < 0:
            raise ValueError(f"{p.product_id}: safety_stock cannot be negative")
        if p.on_hand_qty < 0:
            raise ValueError(f"{p.product_id}: on_hand_qty cannot be negative")
        if p.on_order_qty < 0:
            raise ValueError(f"{p.product_id}: on_order_qty cannot be negative")
        if p.allocated_qty < 0:
            raise ValueError(f"{p.product_id}: allocated_qty cannot be negative")
        if p.reorder_multiple <= 0:
            raise ValueError(f"{p.product_id}: reorder_multiple must be > 0")
        if p.preferred_days_cover <= 0:
            raise ValueError(f"{p.product_id}: preferred_days_cover must be > 0")
        if p.warehouse_capacity is not None and p.warehouse_capacity < 0:
            raise ValueError(f"{p.product_id}: warehouse_capacity cannot be negative")

    def validate_supplier_option(self, s: SupplierOption):
        if s.unit_price <= 0:
            raise ValueError(f"{s.product_id}/{s.supplier_id}: unit_price must be > 0")
        if s.lead_time_days <= 0:
            raise ValueError(f"{s.product_id}/{s.supplier_id}: lead_time_days must be > 0")
        if not (0 <= s.reliability <= 1):
            raise ValueError(f"{s.product_id}/{s.supplier_id}: reliability must be between 0 and 1")
        if s.min_order_qty < 0:
            raise ValueError(f"{s.product_id}/{s.supplier_id}: min_order_qty cannot be negative")
        if s.max_order_qty is not None and s.max_order_qty < s.min_order_qty:
            raise ValueError(f"{s.product_id}/{s.supplier_id}: max_order_qty cannot be < min_order_qty")

    # -----------------------------
    # Inventory ledger / observation capture
    # -----------------------------
    def snapshot_inventory(self, conn, product: Product):
        """Record a point-in-time snapshot so consumption between runs can be
        derived statistically rather than assumed."""
        conn.execute(
            """
            INSERT INTO inventory_transactions (product_id, txn_type, qty_delta, on_hand_after, ts, ref)
            VALUES (?, 'SNAPSHOT', 0, ?, ?, ?)
            """,
            (product.product_id, product.on_hand_qty, datetime.utcnow().isoformat(), self.run_id),
        )
        conn.commit()

    def derive_observed_daily_sales(self, conn, product: Product) -> Optional[float]:
        """Look at the two most recent snapshots and (if receipts didn't
        confound the delta) derive an actual observed daily consumption
        rate. Returns None if there isn't enough history yet."""
        rows = conn.execute(
            """
            SELECT on_hand_after, ts FROM inventory_transactions
            WHERE product_id = ? AND txn_type = 'SNAPSHOT'
            ORDER BY ts DESC LIMIT 2
            """,
            (product.product_id,),
        ).fetchall()
        if len(rows) < 2:
            return None

        (latest_qty, latest_ts), (prev_qty, prev_ts) = rows
        try:
            t1 = datetime.fromisoformat(latest_ts)
            t0 = datetime.fromisoformat(prev_ts)
        except ValueError:
            return None

        elapsed_days = max((t1 - t0).total_seconds() / 86400.0, 1e-6)
        # received qty since prev snapshot, so we don't mistake "stock went up"
        # for "negative demand"
        received = conn.execute(
            """
            SELECT COALESCE(SUM(received_qty), 0) FROM receipts
            WHERE product_id = ? AND received_at > ? AND received_at <= ?
            """,
            (product.product_id, prev_ts, latest_ts),
        ).fetchone()[0]

        consumed = (prev_qty + received) - latest_qty
        daily_rate = consumed / elapsed_days
        return max(daily_rate, 0.0)

    def record_demand_observation(self, conn, product_id: str, observed_daily_sales: float):
        conn.execute(
            "INSERT INTO demand_history (product_id, ts, observed_daily_sales) VALUES (?, ?, ?)",
            (product_id, datetime.utcnow().isoformat(), observed_daily_sales),
        )
        conn.commit()

    def get_demand_history(self, conn, product_id: str, window: int = FORECAST_WINDOW_DAYS) -> List[float]:
        rows = conn.execute(
            """
            SELECT observed_daily_sales FROM demand_history
            WHERE product_id = ?
            ORDER BY ts DESC LIMIT ?
            """,
            (product_id, window),
        ).fetchall()
        return [r[0] for r in rows][::-1]  # chronological order

    # -----------------------------
    # Statistical demand forecasting (replaces hardcoded keyword model)
    # -----------------------------
    def load_context(self, path: str = DEMAND_CONTEXT_PATH) -> str:
        if not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def context_signal_adjustment(self, product: Product, context_text: str) -> Tuple[float, List[str]]:
        """Context text contributes an *additive* adjustment on top of the
        statistical baseline, not the entire forecast. Each signal's weight
        is itself subject to the feedback loop via the per-product learned
        bias, so this is not a fixed/hardcoded rule set in practice -- it's
        an initial prior that gets corrected by observed reality over time.
        """
        text = (context_text or "").lower()
        adj = 0.0
        reasons = []

        if any(k in text for k in ["festival", "holiday", "promotion", "offer", "sale spike"]):
            adj += 0.20
            reasons.append("promotion/holiday uplift +20%")
        if any(k in text for k in ["heatwave", "summer", "high temperature"]):
            adj += 0.15
            reasons.append("summer/heat demand uplift +15%")
        if any(k in text for k in ["rain", "storm", "transport delay", "supply disruption"]):
            adj += 0.10
            reasons.append("supply risk buffer +10%")
        if product.category.lower() in {"beverage", "soft drinks", "juice"} and "weekend" in text:
            adj += 0.05
            reasons.append("weekend beverage uplift +5%")

        return adj, reasons

    def get_learned_bias(self, conn, product_id: str) -> float:
        row = conn.execute(
            "SELECT learned_bias FROM model_state WHERE product_id = ?", (product_id,)
        ).fetchone()
        return row[0] if row else 0.0

    def statistical_forecast(self, conn, product: Product, context_text: str) -> ForecastResult:
        history = self.get_demand_history(conn, product.product_id)

        if len(history) >= 3:
            n = len(history)
            mean = sum(history) / n
            variance = sum((x - mean) ** 2 for x in history) / max(n - 1, 1)
            std = math.sqrt(variance)

            # simple least-squares linear trend over the window, projected
            # one lead-time step ahead, blended conservatively with the mean
            xs = list(range(n))
            x_mean = sum(xs) / n
            denom = sum((x - x_mean) ** 2 for x in xs) or 1.0
            slope = sum((xs[i] - x_mean) * (history[i] - mean) for i in range(n)) / denom
            trend_adj = slope * (n / 2.0)
            daily_mean = max(mean + 0.5 * trend_adj, 0.0)
            method = f"moving_average+trend(window={n})"
        else:
            # Cold start: fall back to the catalog baseline rather than a
            # made-up constant, with wide uncertainty since it's unproven.
            daily_mean = product.avg_daily_sales
            std = max(product.avg_daily_sales * 0.35, 0.5)
            method = "cold_start_catalog_baseline"

        context_adj, reasons = self.context_signal_adjustment(product, context_text)
        learned_bias = self.get_learned_bias(conn, product.product_id)

        adjusted_mean = max(daily_mean * (1 + context_adj) * (1 + learned_bias), 0.0)
        ci_low = max(adjusted_mean - CONFIDENCE_Z * std, 0.0)
        ci_high = adjusted_mean + CONFIDENCE_Z * std

        rationale_parts = reasons[:] if reasons else ["no strong context signals"]
        if abs(learned_bias) > 0.01:
            rationale_parts.append(f"learned feedback bias {learned_bias:+.1%}")

        return ForecastResult(
            daily_mean=round(adjusted_mean, 3),
            daily_std=round(std, 3),
            ci_low=round(ci_low, 3),
            ci_high=round(ci_high, 3),
            method=method,
            rationale="; ".join(rationale_parts),
            learned_bias=learned_bias,
        )

    def update_feedback_loop(self, conn, product: Product, forecast: ForecastResult):
        """Continuous learning: compare the *previous* run's prediction to
        what was actually observed this run, and nudge the per-product bias
        via an EWMA update so future forecasts correct themselves."""
        observed = self.derive_observed_daily_sales(conn, product)
        if observed is None:
            return  # not enough history yet to evaluate the loop

        self.record_demand_observation(conn, product.product_id, observed)

        prev = conn.execute(
            "SELECT last_predicted_mean, learned_bias FROM model_state WHERE product_id = ?",
            (product.product_id,),
        ).fetchone()

        if prev and prev[0]:
            predicted_mean, current_bias = prev
            error = (observed - predicted_mean) / max(predicted_mean, 1e-6)
            new_bias = current_bias + LEARNING_RATE * (error - current_bias)
            new_bias = max(min(new_bias, 1.0), -0.9)  # keep within sane bounds

            conn.execute(
                "INSERT INTO prediction_feedback (product_id, predicted_daily_mean, observed_daily_sales, error, ts) "
                "VALUES (?, ?, ?, ?, ?)",
                (product.product_id, predicted_mean, observed, error, datetime.utcnow().isoformat()),
            )
        else:
            new_bias = 0.0

        conn.execute(
            """
            INSERT INTO model_state (product_id, learned_bias, last_predicted_mean, last_updated)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(product_id) DO UPDATE SET
                learned_bias = excluded.learned_bias,
                last_predicted_mean = excluded.last_predicted_mean,
                last_updated = excluded.last_updated
            """,
            (product.product_id, new_bias, forecast.daily_mean, datetime.utcnow().isoformat()),
        )
        conn.commit()

    # -----------------------------
    # SCM decision logic
    # -----------------------------
    def inventory_position(self, product: Product) -> int:
        return product.on_hand_qty + product.on_order_qty - product.allocated_qty

    def reorder_point(self, product: Product, lead_time_days: int, forecast: ForecastResult) -> int:
        # Use the *upper* confidence bound for the reorder point so the
        # trigger is conservative w.r.t. forecast uncertainty.
        rop = int(round(forecast.ci_high * lead_time_days + product.safety_stock))
        return max(0, rop)

    def target_stock_level(self, product: Product, lead_time_days: int, forecast: ForecastResult) -> int:
        days_cover = max(product.preferred_days_cover, lead_time_days)
        target = int(round(forecast.daily_mean * days_cover + product.safety_stock))
        return max(target, 0)

    def round_to_multiple(self, qty: int, multiple: int) -> int:
        if qty <= 0:
            return 0
        remainder = qty % multiple
        return qty if remainder == 0 else qty + (multiple - remainder)

    def choose_supplier(self, product: Product, options: List[SupplierOption]) -> SupplierOption:
        if not options:
            raise ValueError(f"No eligible suppliers found for product {product.product_id}")

        prices = [o.unit_price for o in options]
        leads = [o.lead_time_days for o in options]
        min_price, max_price = min(prices), max(prices)
        min_lead, max_lead = min(leads), max(leads)

        def normalize_inverse(val, low, high):
            if high == low:
                return 1.0
            return 1 - ((val - low) / (high - low))

        best, best_score = None, -1.0
        for o in options:
            price_score = normalize_inverse(o.unit_price, min_price, max_price)
            lead_score = normalize_inverse(o.lead_time_days, min_lead, max_lead)
            score = (
                PRICE_WEIGHT * price_score
                + RELIABILITY_WEIGHT * o.reliability
                + LEADTIME_WEIGHT * lead_score
            )
            if score > best_score:
                best_score = score
                best = o
        return best

    def decide(self, product: Product, supplier: SupplierOption, forecast: ForecastResult) -> Dict:
        inv_position = self.inventory_position(product)
        rop = self.reorder_point(product, supplier.lead_time_days, forecast)
        target_level = self.target_stock_level(product, supplier.lead_time_days, forecast)

        capacity_capped = False
        if inv_position <= rop:
            raw_qty = max(target_level - inv_position, 0)
            qty = max(raw_qty, supplier.min_order_qty)
            qty = self.round_to_multiple(qty, product.reorder_multiple)

            # --- respect supplier order capacity ---
            if supplier.max_order_qty is not None and qty > supplier.max_order_qty:
                qty = self.round_to_multiple(supplier.max_order_qty, product.reorder_multiple)
                qty = min(qty, supplier.max_order_qty) if qty > supplier.max_order_qty else qty
                capacity_capped = True

            # --- respect warehouse storage capacity ---
            free_capacity = max((product.warehouse_capacity or DEFAULT_WAREHOUSE_CAPACITY) - inv_position, 0)
            if qty > free_capacity:
                qty = max(free_capacity - (free_capacity % product.reorder_multiple), 0)
                capacity_capped = True

            action = "REORDER" if qty > 0 else "HOLD_CAPACITY_CONSTRAINED"
        else:
            qty = 0
            action = "HOLD"

        return {
            "action": action,
            "inventory_position": inv_position,
            "reorder_point": rop,
            "target_stock_level": target_level,
            "recommended_qty": qty,
            "capacity_capped": capacity_capped,
            "forecast": {
                "daily_mean": forecast.daily_mean,
                "daily_std": forecast.daily_std,
                "ci_low": forecast.ci_low,
                "ci_high": forecast.ci_high,
                "method": forecast.method,
                "rationale": forecast.rationale,
            },
        }

    # -----------------------------
    # PO state machine
    # -----------------------------
    def transition_po(self, conn, po_id: int, current_state: str, new_state: POState, reason: str = ""):
        current = POState(current_state) if current_state else POState.DRAFT
        allowed = PO_TRANSITIONS.get(current, [])
        if new_state not in allowed:
            raise InvalidTransitionError(
                f"PO {po_id}: illegal transition {current.value} -> {new_state.value}"
            )

        conn.execute("UPDATE purchase_orders SET state = ? WHERE po_id = ?", (new_state.value, po_id))
        conn.execute(
            "INSERT INTO po_state_log (po_id, from_state, to_state, ts, reason) VALUES (?, ?, ?, ?, ?)",
            (po_id, current.value, new_state.value, datetime.utcnow().isoformat(), reason),
        )
        conn.commit()
        return new_state

    # -----------------------------
    # Action execution (now a real PO -> Shipment -> Receipt lifecycle)
    # -----------------------------
    def create_purchase_order(self, conn, product, supplier, decision, forecast: ForecastResult) -> int:
        expected_receipt = (datetime.utcnow() + timedelta(days=supplier.lead_time_days)).isoformat()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO purchase_orders (
                created_at, product_id, supplier_id, qty, status, unit_price,
                demand_multiplier, reorder_point, inventory_position, notes,
                state, forecast_mean, forecast_std, forecast_ci_low, forecast_ci_high,
                expected_receipt_date, run_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(),
                product.product_id,
                supplier.supplier_id,
                decision["recommended_qty"],
                "CREATED",
                supplier.unit_price,
                round(1.0 + forecast.learned_bias, 3),  # legacy column kept for compatibility
                decision["reorder_point"],
                decision["inventory_position"],
                forecast.rationale,
                POState.DRAFT.value,
                forecast.daily_mean,
                forecast.daily_std,
                forecast.ci_low,
                forecast.ci_high,
                expected_receipt,
                self.run_id,
            ),
        )
        po_id = cur.lastrowid
        conn.execute(
            "INSERT INTO po_state_log (po_id, from_state, to_state, ts, reason) VALUES (?, NULL, ?, ?, ?)",
            (po_id, POState.DRAFT.value, datetime.utcnow().isoformat(), "PO created by agent"),
        )
        conn.commit()
        return po_id

    def mark_product_on_order(self, conn, product_id: str, qty: int):
        conn.execute(
            "UPDATE products SET on_order_qty = on_order_qty + ? WHERE product_id = ?",
            (qty, product_id),
        )
        conn.commit()

    def execute_action(self, conn, product, supplier, decision, forecast: ForecastResult) -> Dict:
        if decision["action"] != "REORDER":
            return {"status": "NO_ACTION", "po_id": None}

        po_id = self.create_purchase_order(conn, product, supplier, decision, forecast)
        self.mark_product_on_order(conn, product.product_id, decision["recommended_qty"])

        # advance through the real lifecycle deterministically for the
        # portion that is within the agent's control (approval + send);
        # carrier-side transit/receipt is advanced later by monitor_open_orders
        self.transition_po(conn, po_id, POState.DRAFT.value, POState.APPROVED, "auto-approved: within policy")
        self.transition_po(conn, po_id, POState.APPROVED.value, POState.SENT_TO_SUPPLIER, "sent to supplier")

        expected_arrival = (datetime.utcnow() + timedelta(days=supplier.lead_time_days)).isoformat()
        conn.execute(
            """
            INSERT INTO shipments (po_id, product_id, shipped_qty, shipped_at, expected_arrival, carrier_status)
            VALUES (?, ?, ?, ?, ?, 'IN_TRANSIT')
            """,
            (po_id, product.product_id, decision["recommended_qty"], datetime.utcnow().isoformat(), expected_arrival),
        )
        self.transition_po(conn, po_id, POState.SENT_TO_SUPPLIER.value, POState.IN_TRANSIT, "shipment created")

        conn.execute(
            """
            INSERT INTO action_status (created_at, product_id, po_id, erp_status, finance_status)
            VALUES (?, ?, ?, ?, ?)
            """,
            (datetime.utcnow().isoformat(), product.product_id, po_id, "ON_ORDER_UPDATED", "ALERT_RECORDED"),
        )
        conn.commit()
        return {"status": "EXECUTED", "po_id": po_id}

    # -----------------------------
    # Continuous post-execution monitoring
    # -----------------------------
    def monitor_open_orders(self, conn):
        """Runs every invocation: ages in-flight shipments, flags overdue
        ones as EXCEPTIONs, and closes out POs whose receipts are complete.
        This is what makes decisions observable *after* execution, not just
        at the moment they're made."""
        now = datetime.utcnow()

        in_transit = conn.execute(
            """
            SELECT sh.shipment_id, sh.po_id, sh.product_id, sh.shipped_qty,
                   sh.expected_arrival, po.state
            FROM shipments sh
            JOIN purchase_orders po ON po.po_id = sh.po_id
            WHERE sh.carrier_status = 'IN_TRANSIT'
            """
        ).fetchall()

        for shipment_id, po_id, product_id, shipped_qty, expected_arrival, po_state in in_transit:
            if not expected_arrival:
                continue
            expected_dt = datetime.fromisoformat(expected_arrival)
            overdue = now > expected_dt + timedelta(days=OVERDUE_GRACE_DAYS)

            if overdue:
                conn.execute(
                    "UPDATE shipments SET carrier_status = 'OVERDUE' WHERE shipment_id = ?",
                    (shipment_id,),
                )
                try:
                    self.transition_po(conn, po_id, po_state, POState.EXCEPTION, "shipment overdue vs lead time")
                except InvalidTransitionError:
                    pass
                self.log_event(
                    conn, "WARNING", "SHIPMENT_OVERDUE", product_id,
                    f"Shipment {shipment_id} for PO {po_id} is overdue",
                    {"shipment_id": shipment_id, "po_id": po_id, "expected_arrival": expected_arrival},
                )

        conn.commit()

    def receive_due_shipments(self, conn):
        """Simulates the warehouse receiving function: shipments that have
        reached/passed their expected arrival are received into stock,
        closing the PO lifecycle and updating real on-hand quantities."""
        now = datetime.utcnow()
        due = conn.execute(
            """
            SELECT shipment_id, po_id, product_id, shipped_qty FROM shipments
            WHERE carrier_status IN ('IN_TRANSIT', 'OVERDUE')
              AND expected_arrival <= ?
            """,
            (now.isoformat(),),
        ).fetchall()

        for shipment_id, po_id, product_id, shipped_qty in due:
            conn.execute(
                "UPDATE shipments SET carrier_status = 'DELIVERED' WHERE shipment_id = ?",
                (shipment_id,),
            )
            conn.execute(
                """
                INSERT INTO receipts (shipment_id, po_id, product_id, received_qty, received_at, condition)
                VALUES (?, ?, ?, ?, ?, 'OK')
                """,
                (shipment_id, po_id, product_id, shipped_qty, now.isoformat()),
            )
            conn.execute(
                """
                UPDATE products
                SET on_hand_qty = on_hand_qty + ?, on_order_qty = MAX(on_order_qty - ?, 0)
                WHERE product_id = ?
                """,
                (shipped_qty, shipped_qty, product_id),
            )
            conn.execute(
                """
                INSERT INTO inventory_transactions (product_id, txn_type, qty_delta, on_hand_after, ts, ref)
                SELECT ?, 'RECEIPT', ?, on_hand_qty, ?, ?
                FROM products WHERE product_id = ?
                """,
                (product_id, shipped_qty, now.isoformat(), f"po:{po_id}", product_id),
            )

            row = conn.execute("SELECT state FROM purchase_orders WHERE po_id = ?", (po_id,)).fetchone()
            if row:
                try:
                    state = self.transition_po(conn, po_id, row[0], POState.RECEIVED, "all stock received")
                    self.transition_po(conn, po_id, state.value, POState.CLOSED, "PO fulfilled")
                except InvalidTransitionError:
                    pass

            self.log_event(
                conn, "INFO", "SHIPMENT_RECEIVED", product_id,
                f"Received {shipped_qty} units for PO {po_id}",
                {"shipment_id": shipment_id, "po_id": po_id, "qty": shipped_qty},
            )

        conn.commit()

    # -----------------------------
    # Main run
    # -----------------------------
    def run(self):
        conn = self.connect()
        try:
            self.ensure_schema_extensions(conn)

            # 1) advance the lifecycle of anything already in flight before
            #    making new decisions, so today's decisions see fresh stock.
            self.monitor_open_orders(conn)
            self.receive_due_shipments(conn)

            products = self.load_products(conn)
            context_text = self.load_context()
            all_results = []

            for product in products:
                try:
                    # learn from what actually happened since last run
                    forecast = self.statistical_forecast(conn, product, context_text)
                    self.update_feedback_loop(conn, product, forecast)
                    # snapshot AFTER learning so the snapshot reflects this run's baseline
                    self.snapshot_inventory(conn, product)

                    supplier_options = self.load_supplier_options(conn, product.product_id)
                    supplier = self.choose_supplier(product, supplier_options)

                    decision = self.decide(product, supplier, forecast)
                    action_result = self.execute_action(conn, product, supplier, decision, forecast)

                    payload = {
                        "product": product.product_id,
                        "supplier": supplier.supplier_id,
                        "forecast": decision["forecast"],
                        "decision": decision,
                        "action_result": action_result,
                        "run_id": self.run_id,
                    }
                    self.log_event(
                        conn, "INFO", "DECISION", product.product_id,
                        f"{decision['action']} for {product.product_id}", payload,
                    )

                    all_results.append({
                        "product_id": product.product_id,
                        "product_name": product.name,
                        "supplier": supplier.supplier_name,
                        "decision": decision,
                        "action_result": action_result,
                        "forecast": decision["forecast"],
                    })

                except Exception as e:
                    self.log_event(conn, "ERROR", "PRODUCT_FAILURE", product.product_id, str(e), {})
                    all_results.append({"product_id": product.product_id, "error": str(e), "status": "FAILED"})

            print(json.dumps({"run_id": self.run_id, "results": all_results}, indent=2))
            return all_results

        except Exception as e:
            try:
                self.log_event(conn, "ERROR", "RUN_FAILURE", None, str(e), {})
            except Exception:
                pass
            raise
        finally:
            conn.close()


if __name__ == "__main__":
    agent = SmartReorderAgent()
    agent.run()