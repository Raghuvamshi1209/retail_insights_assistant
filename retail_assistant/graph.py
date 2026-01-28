from __future__ import annotations
from typing import TypedDict, Any, Dict, List
import pandas as pd
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from .prompts import SYSTEM_GUARDRAILS, PLANNER_INSTRUCTIONS, NARRATOR_INSTRUCTIONS, SUMMARY_INSTRUCTIONS
from .utils import extract_json, summarize_df
from .validators import validate_plan, validate_sql_is_select
from .sql_builder import build_sql
from .llm_providers.gemini_chat import GeminiChat


class AssistantState(TypedDict, total=False):
    user_query: str
    schema: str
    chat_history: List[Dict[str, str]]
    # pass DuckDB connection via state to avoid LangGraph config propagation issues
    duckdb_conn: Any

    plan: Dict[str, Any]
    sql: str
    result_df: pd.DataFrame
    answer: str
    warnings: List[str]
    max_rows: int

    _summary_tables: Dict[str, pd.DataFrame]


def _schema_cols(schema_md: str) -> List[str]:
    cols=[]
    for line in (schema_md or '').splitlines():
        if line.startswith('- '):
            cols.append(line[2:].split(' (dtype=')[0].strip())
    return cols


def planner_node(gemini_model: str, temperature: float):
    chat = GeminiChat(model=gemini_model, temperature=temperature)

    few_shots = [
        '{"q":"Top categories by shipped revenue","hint":"metrics=[shipped_amount], group_by=[Category], sort shipped_amount desc"}',
        '{"q":"Cancellation rate by state","hint":"metrics=[cancel_rate], group_by=[ship-state], sort cancel_rate desc"}',
    ]

    def _node(state: AssistantState, config: RunnableConfig | None = None) -> AssistantState:
        q = state.get('user_query','')
        schema = state.get('schema','')
        history = state.get('chat_history', [])

        messages = [
            {'role':'system','content': SYSTEM_GUARDRAILS},
            {'role':'system','content': 'You are the Planner agent.'},
            {'role':'system','content': PLANNER_INSTRUCTIONS},
            {'role':'system','content': 'SCHEMA\n'+schema},
            {'role':'system','content': 'FEW_SHOT_HINTS\n'+'\n'.join(few_shots)},
        ]
        if history:
            mem='\n'.join([f"{m['role']}: {m['content']}" for m in history])
            messages.append({'role':'system','content': 'RECENT_CHAT\n'+mem})
        messages.append({'role':'user','content': q})

        out = chat.complete(messages, max_output_tokens=800)
        state['plan'] = extract_json(out)
        state.setdefault('warnings', [])
        return state

    return _node


def validator_node():
    def _node(state: AssistantState, config: RunnableConfig | None = None) -> AssistantState:
        schema_cols = _schema_cols(state.get('schema',''))
        plan, warns = validate_plan(state.get('plan') or {}, schema_cols)
        state['plan'] = plan
        state.setdefault('warnings', [])
        state['warnings'].extend(warns)
        return state
    return _node


def extractor_node():
    def _node(state: AssistantState, config: RunnableConfig | None = None) -> AssistantState:
        duckdb_conn = state.get('duckdb_conn')
        if duckdb_conn is None:
            raise RuntimeError('DuckDB connection missing in state. Ensure state["duckdb_conn"] is set in app.py')

        schema_cols = _schema_cols(state.get('schema',''))
        sql = build_sql(state.get('plan') or {}, schema_cols)
        validate_sql_is_select(sql)
        df = duckdb_conn.execute(sql).fetchdf()
        state['sql'] = sql
        state['result_df'] = df
        return state
    return _node


def narrator_node(gemini_model: str, temperature: float):
    chat = GeminiChat(model=gemini_model, temperature=temperature)

    def _node(state: AssistantState, config: RunnableConfig | None = None) -> AssistantState:
        df_summary = summarize_df(state.get('result_df'), max_rows=int(state.get('max_rows') or 8))
        messages = [
            {'role':'system','content': SYSTEM_GUARDRAILS},
            {'role':'system','content': 'You are the Narrator agent.'},
            {'role':'system','content': NARRATOR_INSTRUCTIONS},
            {'role':'system','content': 'USER_QUESTION\n'+state.get('user_query','')},
            {'role':'system','content': 'PLAN_JSON\n'+str(state.get('plan'))},
            {'role':'system','content': 'SQL\n'+state.get('sql','')},
            {'role':'system','content': 'RESULT_SUMMARY\n'+str(df_summary)},
        ]
        state['answer'] = chat.complete(messages, max_output_tokens=700)
        return state

    return _node


def build_chat_graph(gemini_model: str='gemini-2.5-flash', temperature: float=0.1):
    g = StateGraph(AssistantState)
    g.add_node('planner', planner_node(gemini_model, temperature))
    g.add_node('validator', validator_node())
    g.add_node('extractor', extractor_node())
    g.add_node('narrator', narrator_node(gemini_model, temperature))

    g.set_entry_point('planner')
    g.add_edge('planner','validator')
    g.add_edge('validator','extractor')
    g.add_edge('extractor','narrator')
    g.add_edge('narrator', END)

    return g.compile()


# Summarization graph
def summary_extractor_node():
    def _node(state: AssistantState, config: RunnableConfig | None = None) -> AssistantState:
        duckdb_conn = state.get('duckdb_conn')
        if duckdb_conn is None:
            raise RuntimeError('DuckDB connection missing in state. Ensure state["duckdb_conn"] is set in app.py')

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

        kpi = duckdb_conn.execute(kpi_sql).fetchdf()
        trend = duckdb_conn.execute(trend_sql).fetchdf()
        state['_summary_tables'] = {'kpi': kpi, 'trend': trend}
        return state

    return _node


def summary_narrator_node(gemini_model: str, temperature: float):
    chat = GeminiChat(model=gemini_model, temperature=temperature)

    def _node(state: AssistantState, config: RunnableConfig | None = None) -> AssistantState:
        payload = {k: summarize_df(df, max_rows=int(state.get('max_rows') or 8)) for k, df in (state.get('_summary_tables') or {}).items()}
        messages = [
            {'role':'system','content': SYSTEM_GUARDRAILS},
            {'role':'system','content': 'You are the Summarization Narrator agent.'},
            {'role':'system','content': SUMMARY_INSTRUCTIONS},
            {'role':'system','content': 'SUMMARY_TABLES\n'+str(payload)},
        ]
        state['answer'] = chat.complete(messages, max_output_tokens=700)
        return state

    return _node


def build_summary_graph(gemini_model: str='gemini-2.5-flash', temperature: float=0.1):
    g = StateGraph(AssistantState)
    g.add_node('summary_extractor', summary_extractor_node())
    g.add_node('summary_narrator', summary_narrator_node(gemini_model, temperature))
    g.set_entry_point('summary_extractor')
    g.add_edge('summary_extractor','summary_narrator')
    g.add_edge('summary_narrator', END)
    return g.compile()
