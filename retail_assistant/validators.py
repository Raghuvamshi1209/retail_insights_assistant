from __future__ import annotations
from typing import Dict, Any, List, Tuple
from .sql_builder import SAFE_METRICS


def validate_sql_is_select(sql: str) -> None:
    s = (sql or '').strip().lower()
    if not s.startswith('select'):
        raise ValueError('Only SELECT queries are allowed.')
    banned = ['insert', 'update', 'delete', 'drop', 'alter', 'create', 'attach', 'copy', 'pragma']
    if any(b in s for b in banned):
        raise ValueError('Potentially unsafe SQL detected.')


def validate_plan(plan: Dict[str, Any], schema_cols: List[str]) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    if not isinstance(plan, dict):
        warnings.append('Planner did not return JSON; using default plan.')
        plan = {}

    plan['intent'] = 'qa'

    group_by = plan.get('group_by') or []
    if not isinstance(group_by, list):
        group_by = []
    plan['group_by'] = [c for c in group_by if c in schema_cols]

    metrics = plan.get('metrics') or []
    if not isinstance(metrics, list):
        metrics = []
    cleaned_metrics = [m for m in metrics if m in SAFE_METRICS]
    if not cleaned_metrics:
        cleaned_metrics = ['shipped_amount']
    plan['metrics'] = cleaned_metrics

    filters = plan.get('filters') or {}
    if not isinstance(filters, dict):
        filters = {}
    plan['filters'] = {k: v for k, v in filters.items() if k in schema_cols}

    time = plan.get('time') or {}
    if not isinstance(time, dict):
        time = {'from': None, 'to': None}
    time.setdefault('from', None)
    time.setdefault('to', None)
    plan['time'] = time

    try:
        plan['limit'] = int(plan.get('limit') or 10)
    except Exception:
        plan['limit'] = 10
    plan['limit'] = max(1, min(plan['limit'], 200))

    notes = (plan.get('notes') or '').lower()
    if 'yoy' in notes or 'year over year' in notes:
        warnings.append('YoY requires multiple years; this dataset may not support true YoY.')

    return plan, warnings
