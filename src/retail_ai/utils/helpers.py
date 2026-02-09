from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import pandas as pd


def extract_json(text: str):
    if not text:
        raise ValueError("Empty model response")
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("No JSON found in model output")
    return json.loads(m.group(0))


def get_schema_metadata(df: pd.DataFrame):
    cols = [f"- {c} (dtype={df[c].dtype})" for c in df.columns]
    return "\n".join(
        [
            "Dataset columns:",
            *cols,
            "",
            "Notes:",
            "- Date column is MM-DD-YY.",
            "- Use Amount as sales value and Qty as units.",
            "- Status indicates shipped/cancelled/returned states.",
        ]
    )


def df_to_markdown(df: pd.DataFrame, max_rows: int = 10):
    try:
        return df.head(max_rows).to_markdown(index=False)
    except Exception:
        return df.head(max_rows).to_string(index=False)
