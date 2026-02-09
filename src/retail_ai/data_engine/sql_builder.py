from __future__ import annotations

from typing import Any, Dict, List

SAFE_METRICS = {
    "gross_amount": "SUM(COALESCE(Amount, 0))",
    "shipped_amount": "SUM(CASE WHEN Status LIKE 'Shipped%' THEN COALESCE(Amount,0) ELSE 0 END)",
    "cancelled_amount": "SUM(CASE WHEN lower(Status) LIKE '%cancelled%' THEN COALESCE(Amount,0) ELSE 0 END)",
    "orders": 'COUNT(DISTINCT "Order ID")',
    "units": "SUM(COALESCE(Qty,0))",
    "cancel_rate": "AVG(CASE WHEN lower(Status) LIKE '%cancelled%' THEN 1.0 ELSE 0.0 END)",
}


def _sql_literal(v: Any):
    if v is None:
        return "NULL"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v).replace("'", "''")
    return f"'{s}'"


def build_sql(plan: Dict[str, Any], schema_cols: List[str]):
    group_by = plan.get("group_by") or []
    filters = plan.get("filters") or {}
    metrics = plan.get("metrics") or ["shipped_amount"]
    sort = plan.get("sort") or []
    limit = int(plan.get("limit") or 10)

    select_parts: List[str] = []
    group_parts: List[str] = []

    for col in group_by:
        if col in schema_cols:
            select_parts.append(f'"{col}"')
            group_parts.append(f'"{col}"')

    for m in metrics:
        expr = SAFE_METRICS.get(m)
        if expr:
            select_parts.append(f"{expr} AS {m}")

    if not select_parts:
        select_parts = [SAFE_METRICS["shipped_amount"] + " AS shipped_amount"]

    where_parts: List[str] = []
    time = plan.get("time") or {}
    date_from = time.get("from")
    date_to = time.get("to")
    if date_from:
        where_parts.append(f"TRY_STRPTIME(Date, '%m-%d-%y') >= DATE {_sql_literal(date_from)}")
    if date_to:
        where_parts.append(f"TRY_STRPTIME(Date, '%m-%d-%y') <= DATE {_sql_literal(date_to)}")

    for col, val in filters.items():
        if col not in schema_cols:
            continue
        if isinstance(val, list):
            vals = ",".join(_sql_literal(x) for x in val)
            where_parts.append(f'"{col}" IN ({vals})')
        else:
            where_parts.append(f'"{col}" = {_sql_literal(val)}')

    sql = "SELECT " + ", ".join(select_parts) + "\nFROM sales\n"
    if where_parts:
        sql += "WHERE " + " AND ".join(where_parts) + "\n"
    if group_parts:
        sql += "GROUP BY " + ", ".join(group_parts) + "\n"

    if sort:
        order_parts: List[str] = []
        for s in sort:
            by = s.get("by")
            order = (s.get("order") or "desc").lower()
            if by in schema_cols:
                order_parts.append(f'"{by}" {order.upper()}')
            elif by in SAFE_METRICS:
                order_parts.append(f"{by} {order.upper()}")
        if order_parts:
            sql += "ORDER BY " + ", ".join(order_parts) + "\n"

    sql += f"LIMIT {max(1, min(limit, 200))}"
    return sql
