from __future__ import annotations
import re
import json
from typing import Any, Dict
import pandas as pd
import duckdb


def load_csv(uploaded_file) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded_file, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(uploaded_file, low_memory=False, encoding='latin1')


def make_duckdb_conn() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(database=':memory:')


def get_schema_metadata(df: pd.DataFrame) -> str:
    cols = [f"- {c} (dtype={df[c].dtype})" for c in df.columns]
    return "\n".join([
        'Dataset columns:',
        *cols,
        '',
        'Notes:',
        '- Date column is MM-DD-YY.',
        '- Use Amount as sales value and Qty as units.',
        '- Status indicates shipped/cancelled/returned states.'
    ])


def extract_json(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError('Empty model response')
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError('No JSON found in model output')
    return json.loads(m.group(0))


def summarize_df(df: pd.DataFrame, max_rows: int = 8) -> Dict[str, Any]:
    if df is None:
        return {'rows': 0}
    out: Dict[str, Any] = {'rows': int(len(df)), 'columns': list(df.columns)}
    if len(df) > 0:
        out['sample'] = df.head(max_rows).to_dict(orient='records')
    return out
