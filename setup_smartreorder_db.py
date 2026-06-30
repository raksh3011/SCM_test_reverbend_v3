import os
import sqlite3

DB_PATH = "smartreorder.db"

SCHEMA_SQL = """
DROP TABLE IF EXISTS action_status;
DROP TABLE IF EXISTS purchase_orders;
DROP TABLE IF EXISTS audit_log;
DROP TABLE IF EXISTS supplier_products;
DROP TABLE IF EXISTS suppliers;
DROP TABLE IF EXISTS products;

CREATE TABLE products (
    product_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    avg_daily_sales REAL NOT NULL,
    safety_stock INTEGER NOT NULL,
    on_hand_qty INTEGER NOT NULL,
    on_order_qty INTEGER NOT NULL DEFAULT 0,
    allocated_qty INTEGER NOT NULL DEFAULT 0,
    reorder_multiple INTEGER NOT NULL DEFAULT 1,
    preferred_days_cover INTEGER NOT NULL DEFAULT 14
);

CREATE TABLE suppliers (
    supplier_id TEXT PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE supplier_products (
    supplier_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    unit_price REAL NOT NULL,
    lead_time_days INTEGER NOT NULL,
    reliability REAL NOT NULL,
    min_order_qty INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (supplier_id, product_id),
    FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE TABLE purchase_orders (
    po_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    product_id TEXT NOT NULL,
    supplier_id TEXT NOT NULL,
    qty INTEGER NOT NULL,
    status TEXT NOT NULL,
    unit_price REAL NOT NULL,
    demand_multiplier REAL NOT NULL,
    reorder_point INTEGER NOT NULL,
    inventory_position INTEGER NOT NULL,
    notes TEXT
);

CREATE TABLE action_status (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    product_id TEXT NOT NULL,
    po_id INTEGER,
    erp_status TEXT,
    finance_status TEXT
);

CREATE TABLE audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    level TEXT NOT NULL,
    event_type TEXT NOT NULL,
    product_id TEXT,
    message TEXT NOT NULL,
    payload_json TEXT
);
"""

PRODUCTS = [
    ("P1", "Cola 500ml", "beverage", 40, 80, 50, 20, 5, 12, 14),
    ("P2", "Orange Juice 1L", "beverage", 18, 40, 140, 0, 10, 6, 14),
    ("P3", "Energy Drink", "beverage", 25, 60, 15, 0, 0, 24, 21),
]

SUPPLIERS = [
    ("S1", "FreshSource Distributors"),
    ("S2", "QuickSupply Beverages"),
    ("S3", "National Drinks Wholesale"),
]

SUPPLIER_PRODUCTS = [
    # P1
    ("S1", "P1", 18.0, 3, 0.92, 24),
    ("S2", "P1", 17.5, 5, 0.85, 12),
    ("S3", "P1", 18.4, 2, 0.95, 24),

    # P2
    ("S1", "P2", 42.0, 4, 0.90, 12),
    ("S3", "P2", 41.5, 6, 0.88, 12),

    # P3
    ("S2", "P3", 26.0, 2, 0.86, 24),
    ("S3", "P3", 27.5, 1, 0.94, 24),
]


def main():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.executescript(SCHEMA_SQL)

    cur.executemany(
        """
        INSERT INTO products (
            product_id, name, category, avg_daily_sales, safety_stock,
            on_hand_qty, on_order_qty, allocated_qty,
            reorder_multiple, preferred_days_cover
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        PRODUCTS,
    )

    cur.executemany(
        "INSERT INTO suppliers (supplier_id, name) VALUES (?, ?)",
        SUPPLIERS,
    )

    cur.executemany(
        """
        INSERT INTO supplier_products (
            supplier_id, product_id, unit_price, lead_time_days,
            reliability, min_order_qty
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        SUPPLIER_PRODUCTS,
    )

    conn.commit()
    conn.close()
    print(f"Created {DB_PATH} successfully.")


if __name__ == "__main__":
    main()
