from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


@dataclass
class DataLoader:
    """loads the datasets from a folder. Keeps Input Output concerns separate from UI."""

    input_dir: Path

    def list_csv_files(self):
        if not self.input_dir.exists():
            return []
        return sorted(self.input_dir.glob("*.csv"))

    def load(self, path: Path):
        try:
            df = pd.read_csv(path, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, low_memory=False, encoding="latin1")

        # Standard numeric coercions
        for col in ["Amount", "Qty"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
