from __future__ import annotations

import os
from typing import Any, Dict, List, TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

from retail_ai.llm.gemini_client import GeminiChat
from retail_ai.utils.config_loader import load_yaml
from retail_ai.utils.helpers import df_to_markdown


class SummaryState(TypedDict, total=False):
    schema: str
    duckdb_service: Any
    max_rows: int

    answer: str
    _summary_tables: Dict[str, pd.DataFrame]


def build_summary_graph(
    cfg_path: str = "config/model_config.yaml", prompts_path: str = "config/prompt_templates.yaml"
):
    cfg = load_yaml(cfg_path)
    prompts = load_yaml(prompts_path)

    model = os.getenv("GEMINI_MODEL", cfg.get("llm", {}).get("model", "gemini-2.5-flash"))
    temperature = float(os.getenv("TEMPERATURE", cfg.get("llm", {}).get("temperature", 0.1)))
    llm = GeminiChat(model=model, temperature=temperature)

    summary_tokens = int(cfg.get("llm", {}).get("max_output_tokens", {}).get("summary", 1100))

    def extractor(state: SummaryState):
        svc = state.get("duckdb_service")
        if svc is None:
            raise RuntimeError("duckdb_service missing in state")
        kpi_sql = """SELECT
          COUNT(*) AS rows,
          COUNT(DISTINCT \"Order ID\") AS orders,
          SUM(COALESCE(Qty,0)) AS units,
          SUM(COALESCE(Amount,0)) AS gross_amount,
          SUM(CASE WHEN Status LIKE 'Shipped%' THEN COALESCE(Amount,0) ELSE 0 END) AS shipped_amount,
          SUM(CASE WHEN lower(Status) LIKE '%cancelled%' THEN COALESCE(Amount,0) ELSE 0 END) AS cancelled_amount,
          AVG(CASE WHEN lower(Status) LIKE '%cancelled%' THEN 1.0 ELSE 0.0 END) AS cancel_rate,
          MIN(TRY_STRPTIME(Date, '%m-%d-%y')) AS min_date,
          MAX(TRY_STRPTIME(Date, '%m-%d-%y')) AS max_date
        FROM sales;"""

        trend_sql = """SELECT
          STRFTIME(TRY_STRPTIME(Date, '%m-%d-%y'), '%Y-%m') AS month,
          COUNT(DISTINCT \"Order ID\") AS orders,
          SUM(COALESCE(Amount,0)) AS gross_amount,
          SUM(CASE WHEN Status LIKE 'Shipped%' THEN COALESCE(Amount,0) ELSE 0 END) AS shipped_amount,
          AVG(CASE WHEN lower(Status) LIKE '%cancelled%' THEN 1.0 ELSE 0.0 END) AS cancel_rate
        FROM sales
        GROUP BY month
        ORDER BY month;"""

        top_cat_sql = """SELECT Category,
          SUM(CASE WHEN Status LIKE 'Shipped%' THEN COALESCE(Amount,0) ELSE 0 END) AS shipped_amount,
          COUNT(DISTINCT \"Order ID\") AS orders,
          SUM(COALESCE(Qty,0)) AS units
        FROM sales
        GROUP BY Category
        ORDER BY shipped_amount DESC
        LIMIT 10;"""

        top_state_sql = """SELECT \"ship-state\" AS ship_state,
          SUM(CASE WHEN Status LIKE 'Shipped%' THEN COALESCE(Amount,0) ELSE 0 END) AS shipped_amount,
          COUNT(DISTINCT \"Order ID\") AS orders
        FROM sales
        GROUP BY ship_state
        ORDER BY shipped_amount DESC
        LIMIT 10;"""

        svc_sql = """SELECT \"ship-service-level\" AS ship_service_level,
          SUM(CASE WHEN Status LIKE 'Shipped%' THEN COALESCE(Amount,0) ELSE 0 END) AS shipped_amount,
          COUNT(DISTINCT \"Order ID\") AS orders
        FROM sales
        GROUP BY ship_service_level
        ORDER BY shipped_amount DESC
        LIMIT 10;"""

        state["_summary_tables"] = {
            "kpi": svc.query_df(kpi_sql),
            "trend": svc.query_df(trend_sql),
            "top_categories": svc.query_df(top_cat_sql),
            "top_states": svc.query_df(top_state_sql),
            "service_levels": svc.query_df(svc_sql),
        }
        return state

    def narrator(state: SummaryState):
        tables = state.get("_summary_tables") or {}
        max_rows = int(state.get("max_rows") or 10)
        payload = {k: df_to_markdown(v, max_rows=max_rows) for k, v in tables.items()}

        msgs = [
            {"role": "system", "content": prompts.get("system_guardrails", "")},
            {"role": "system", "content": "You are the Summarization Narrator agent."},
            {"role": "system", "content": prompts.get("summary_instructions", "")},
            {"role": "system", "content": "SUMMARY_TABLES_MARKDOWN\n" + str(payload)},
        ]
        state["answer"] = llm.complete(msgs, max_output_tokens=summary_tokens)
        return state

    g = StateGraph(SummaryState)
    g.add_node("summary_extractor", extractor)
    g.add_node("summary_narrator", narrator)
    g.set_entry_point("summary_extractor")
    g.add_edge("summary_extractor", "summary_narrator")
    g.add_edge("summary_narrator", END)
    return g.compile()
