"""Unit tests for the pure decision-logic pieces of smart_reorder_agent.py.

Deliberately DB-free: reorder_point, round_to_multiple, choose_supplier,
escalation_reasons and target_stock_level are all pure functions of their
arguments, so they're tested directly without spinning up sqlite fixtures.
The PO state machine is tested against an in-memory sqlite connection since
transition_po's job is specifically to persist + validate transitions.
"""
import sqlite3

import pytest

from smart_reorder_agent import (
    ForecastResult,
    InvalidTransitionError,
    POState,
    Product,
    SmartReorderAgent,
    SupplierOption,
)


def make_product(**overrides):
    defaults = dict(
        product_id="P1", name="Widget", category="general", avg_daily_sales=10.0,
        safety_stock=20, on_hand_qty=50, on_order_qty=0, allocated_qty=0,
        reorder_multiple=10, preferred_days_cover=14, warehouse_capacity=100_000,
    )
    defaults.update(overrides)
    return Product(**defaults)


def make_supplier(**overrides):
    defaults = dict(
        supplier_id="S1", supplier_name="Acme", product_id="P1", unit_price=5.0,
        lead_time_days=7, reliability=0.9, min_order_qty=10, max_order_qty=None,
        lead_time_days_std=0.0,
    )
    defaults.update(overrides)
    return SupplierOption(**defaults)


def make_forecast(**overrides):
    defaults = dict(daily_mean=10.0, daily_std=2.0, ci_low=6.08, ci_high=13.92,
                     method="test", rationale="test")
    defaults.update(overrides)
    return ForecastResult(**defaults)


@pytest.fixture
def agent():
    return SmartReorderAgent(db_path=":memory:")


class TestReorderPoint:
    def test_increases_with_lead_time_variance(self, agent):
        """The specific behavior ds-repl-001 exists to check: two suppliers
        identical except for lead_time_days_std must not produce the same
        reorder point -- the more variable one must trigger reorder sooner."""
        product = make_product()
        forecast = make_forecast()
        low_variance = make_supplier(lead_time_days_std=0.5)
        high_variance = make_supplier(lead_time_days_std=5.0)

        rop_low = agent.reorder_point(product, low_variance, forecast)
        rop_high = agent.reorder_point(product, high_variance, forecast)

        assert rop_high > rop_low

    def test_never_negative(self, agent):
        product = make_product(safety_stock=0)
        forecast = make_forecast(daily_mean=0.0, daily_std=0.0, ci_low=0.0, ci_high=0.0)
        supplier = make_supplier(lead_time_days=1, lead_time_days_std=0.0)
        assert agent.reorder_point(product, supplier, forecast) >= 0


class TestRoundToMultiple:
    def test_rounds_up_to_next_multiple(self, agent):
        assert agent.round_to_multiple(23, 10) == 30

    def test_exact_multiple_unchanged(self, agent):
        assert agent.round_to_multiple(30, 10) == 30

    def test_non_positive_qty_is_zero(self, agent):
        assert agent.round_to_multiple(0, 10) == 0
        assert agent.round_to_multiple(-5, 10) == 0


class TestChooseSupplier:
    def test_prefers_cheaper_more_reliable_faster_supplier(self, agent):
        product = make_product()
        cheap_reliable = make_supplier(supplier_id="S1", unit_price=1.0, reliability=0.99, lead_time_days=2)
        expensive_unreliable = make_supplier(supplier_id="S2", unit_price=10.0, reliability=0.5, lead_time_days=20)
        chosen = agent.choose_supplier(product, [cheap_reliable, expensive_unreliable])
        assert chosen.supplier_id == "S1"

    def test_raises_when_no_suppliers(self, agent):
        with pytest.raises(ValueError):
            agent.choose_supplier(make_product(), [])


class TestEscalationReasons:
    def test_no_escalation_for_small_reliable_order(self, agent):
        product = make_product()
        supplier = make_supplier(unit_price=1.0, reliability=0.95)
        forecast = make_forecast(daily_mean=10.0, ci_low=9.0, ci_high=11.0)
        reasons = agent.escalation_reasons(product, supplier, forecast, qty=10, capacity_capped=False)
        assert reasons == []

    def test_escalates_on_large_order_value(self, agent):
        product = make_product()
        supplier = make_supplier(unit_price=1000.0, reliability=0.95)
        forecast = make_forecast(daily_mean=10.0, ci_low=9.0, ci_high=11.0)
        reasons = agent.escalation_reasons(product, supplier, forecast, qty=1000, capacity_capped=False)
        assert any("order value" in r for r in reasons)

    def test_escalates_on_low_supplier_reliability(self, agent):
        product = make_product()
        supplier = make_supplier(unit_price=1.0, reliability=0.2)
        forecast = make_forecast(daily_mean=10.0, ci_low=9.0, ci_high=11.0)
        reasons = agent.escalation_reasons(product, supplier, forecast, qty=10, capacity_capped=False)
        assert any("reliability" in r for r in reasons)

    def test_escalates_on_capacity_capped(self, agent):
        product = make_product()
        supplier = make_supplier(unit_price=1.0, reliability=0.95)
        forecast = make_forecast(daily_mean=10.0, ci_low=9.0, ci_high=11.0)
        reasons = agent.escalation_reasons(product, supplier, forecast, qty=10, capacity_capped=True)
        assert any("capacity" in r for r in reasons)

    def test_escalates_on_wide_forecast_uncertainty(self, agent):
        product = make_product()
        supplier = make_supplier(unit_price=1.0, reliability=0.95)
        forecast = make_forecast(daily_mean=10.0, ci_low=1.0, ci_high=19.0)  # spans 180% of mean
        reasons = agent.escalation_reasons(product, supplier, forecast, qty=10, capacity_capped=False)
        assert any("confidence interval" in r for r in reasons)


class TestDecide:
    def test_holds_when_inventory_position_above_reorder_point(self, agent):
        product = make_product(on_hand_qty=10_000)
        supplier = make_supplier()
        forecast = make_forecast()
        decision = agent.decide(product, supplier, forecast)
        assert decision["action"] == "HOLD"
        assert decision["recommended_qty"] == 0

    def test_reorders_when_below_reorder_point(self, agent):
        product = make_product(on_hand_qty=5, safety_stock=20)
        supplier = make_supplier()
        forecast = make_forecast()
        decision = agent.decide(product, supplier, forecast)
        assert decision["action"] == "REORDER"
        assert decision["recommended_qty"] > 0

    def test_carries_traceability_fields(self, agent):
        product = make_product(on_hand_qty=5)
        supplier = make_supplier()
        forecast = make_forecast()
        decision = agent.decide(product, supplier, forecast)
        assert "lot_id" in decision["carried_fields"]
        assert "product_identifier" in decision["carried_fields"]
        assert decision["lot_id"]
        assert decision["product_identifier"] == product.product_id


class TestPOStateMachine:
    def _conn(self, agent):
        conn = sqlite3.connect(":memory:")
        conn.executescript(
            """
            CREATE TABLE purchase_orders (po_id INTEGER PRIMARY KEY, state TEXT);
            CREATE TABLE po_state_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT, po_id INTEGER, from_state TEXT,
                to_state TEXT, ts TEXT, reason TEXT
            );
            """
        )
        conn.execute("INSERT INTO purchase_orders (po_id, state) VALUES (1, 'DRAFT')")
        conn.commit()
        return conn

    def test_valid_transition_succeeds(self, agent):
        conn = self._conn(agent)
        new_state = agent.transition_po(conn, 1, "DRAFT", POState.APPROVED, "test")
        assert new_state == POState.APPROVED
        row = conn.execute("SELECT state FROM purchase_orders WHERE po_id = 1").fetchone()
        assert row[0] == "APPROVED"

    def test_illegal_transition_raises(self, agent):
        conn = self._conn(agent)
        with pytest.raises(InvalidTransitionError):
            agent.transition_po(conn, 1, "DRAFT", POState.RECEIVED, "skip states")

    def test_terminal_state_has_no_outgoing_transitions(self, agent):
        conn = self._conn(agent)
        with pytest.raises(InvalidTransitionError):
            agent.transition_po(conn, 1, "CLOSED", POState.DRAFT, "resurrect")
