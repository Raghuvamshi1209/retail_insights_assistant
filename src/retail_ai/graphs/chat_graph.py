from __future__ import annotations

import os
from typing import Any, Dict, List, TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

from retail_ai.llm.gemini_client import GeminiChat
from retail_ai.utils.config_loader import load_yaml
from retail_ai.utils.helpers import df_to_markdown, extract_json
from retail_ai.data_engine.sql_builder import build_sql
from retail_ai.utils.validators import validate_plan, validate_sql_is_select


class ChatState(TypedDict, total=False):
    user_query: str
    schema: str
    chat_history: List[Dict[str, str]]
    duckdb_service: Any  # DuckDBService
    max_rows: int

    plan: Dict[str, Any]
    sql: str
    result_df: pd.DataFrame
    answer: str
    warnings: List[str]


def _schema_cols(schema_md: str):
    cols: List[str] = []
    for line in (schema_md or "").splitlines():
        if line.startswith("- "):
            cols.append(line[2:].split(" (dtype=")[0].strip())
    return cols


def build_chat_graph(
    cfg_path: str = "config/model_config.yaml", prompts_path: str = "config/prompt_templates.yaml"
):
    cfg = load_yaml(cfg_path)
    prompts = load_yaml(prompts_path)

    model = os.getenv("GEMINI_MODEL", cfg.get("llm", {}).get("model", "gemini-2.5-flash"))
    temperature = float(os.getenv("TEMPERATURE", cfg.get("llm", {}).get("temperature", 0.1)))

    planner_tokens = int(cfg.get("llm", {}).get("max_output_tokens", {}).get("planner", 900))
    narrator_tokens = int(cfg.get("llm", {}).get("max_output_tokens", {}).get("narrator", 900))

    llm = GeminiChat(model=model, temperature=temperature)

    few_shots = [
        '{"q":"Top categories by shipped revenue","hint":"metrics=[shipped_amount], group_by=[Category], sort shipped_amount desc"}',
        '{"q":"Cancellation rate by state","hint":"metrics=[cancel_rate], group_by=[ship-state], sort cancel_rate desc"}',
        '{"q":"Service level comparison","hint":"metrics=[shipped_amount], group_by=[ship-service-level]"}',
    ]

    def planner(state: ChatState):
        msgs = [
            {"role": "system", "content": prompts.get("system_guardrails", "")},
            {"role": "system", "content": "You are the Planner agent."},
            {"role": "system", "content": prompts.get("planner_instructions", "")},
            {"role": "system", "content": "SCHEMA\n" + (state.get("schema") or "")},
            {"role": "system", "content": "FEW_SHOT_HINTS\n" + "\n".join(few_shots)},
            {"role": "user", "content": state.get("user_query") or ""},
        ]
        out = llm.complete(msgs, max_output_tokens=planner_tokens)
        state["plan"] = extract_json(out)
        state.setdefault("warnings", [])
        return state

    def validator(state: ChatState):
        schema_cols = _schema_cols(state.get("schema") or "")
        plan, warns = validate_plan(state.get("plan") or {}, schema_cols)
        state["plan"] = plan
        state.setdefault("warnings", []).extend(warns)
        return state

    def extractor(state: ChatState):
        svc = state.get("duckdb_service")
        if svc is None:
            raise RuntimeError("duckdb_service missing in state")
        schema_cols = _schema_cols(state.get("schema") or "")
        sql = build_sql(state.get("plan") or {}, schema_cols)
        validate_sql_is_select(sql)
        df = svc.query_df(sql)
        state["sql"] = sql
        state["result_df"] = df
        return state

    def narrator(state: ChatState):
        df = state.get("result_df")
        max_rows = int(state.get("max_rows") or 10)
        md = df_to_markdown(df, max_rows=max_rows) if isinstance(df, pd.DataFrame) else str(df)
        msgs = [
            {"role": "system", "content": prompts.get("system_guardrails", "")},
            {"role": "system", "content": "You are the Narrator agent."},
            {"role": "system", "content": prompts.get("narrator_instructions", "")},
            {"role": "system", "content": "USER_QUESTION\n" + (state.get("user_query") or "")},
            {"role": "system", "content": "PLAN_JSON\n" + str(state.get("plan"))},
            {"role": "system", "content": "SQL\n" + (state.get("sql") or "")},
            {"role": "system", "content": "RESULT_TABLE\n" + md},
        ]
        state["answer"] = llm.complete(msgs, max_output_tokens=narrator_tokens)
        return state

    g = StateGraph(ChatState)
    g.add_node("planner", planner)
    g.add_node("validator", validator)
    g.add_node("extractor", extractor)
    g.add_node("narrator", narrator)
    g.set_entry_point("planner")
    g.add_edge("planner", "validator")
    g.add_edge("validator", "extractor")
    g.add_edge("extractor", "narrator")
    g.add_edge("narrator", END)
    return g.compile()
