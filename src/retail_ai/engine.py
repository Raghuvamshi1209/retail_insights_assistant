from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from retail_ai.graphs.chat_graph import build_chat_graph
from retail_ai.graphs.summary_graph import build_summary_graph
from retail_ai.utils.helpers import get_schema_metadata
from retail_ai.data_engine.duckdb_service import DuckDBService


@dataclass
class RetailAssistantEngine:
    """High-level façade to run chat and summary flows.

    This class wires together:
    - DuckDBService (deterministic analytics)
    - LangGraph pipelines
    - Schema metadata

    Keeping orchestration here improves readability and testability.
    """

    cfg_path: str = "config/model_config.yaml"
    prompts_path: str = "config/prompt_templates.yaml"

    def __post_init__(self):
        self._chat_graph = build_chat_graph(self.cfg_path, self.prompts_path)
        self._summary_graph = build_summary_graph(self.cfg_path, self.prompts_path)

    def summarize(self, df: pd.DataFrame, max_rows: int = 50):
        svc = DuckDBService.in_memory()
        svc.register_sales(df)
        schema_md = get_schema_metadata(df)
        state = {"schema": schema_md, "duckdb_service": svc, "max_rows": int(max_rows)}
        return self._summary_graph.invoke(state)

    def answer(
        self,
        df: pd.DataFrame,
        question: str,
        chat_history: List[Dict[str, str]] | None = None,
        max_rows: int = 50,
    ):
        svc = DuckDBService.in_memory()
        svc.register_sales(df)
        schema_md = get_schema_metadata(df)
        state = {
            "user_query": question,
            "schema": schema_md,
            "chat_history": chat_history or [],
            "duckdb_service": svc,
            "max_rows": int(max_rows),
        }
        return self._chat_graph.invoke(state)
