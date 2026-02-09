from __future__ import annotations

from dataclasses import dataclass

import duckdb
import pandas as pd


@dataclass
class DuckDBService:
    """Encapsulates the DuckDB connection + data registration."""

    conn: duckdb.DuckDBPyConnection
    table_name: str = "sales"

    @classmethod
    def in_memory(cls, table_name: str = "sales"):
        conn = duckdb.connect(database=":memory:")
        return cls(conn=conn, table_name=table_name)

    def register_sales(self, df: pd.DataFrame):
        self.conn.register("sales_df", df)
        self.conn.execute(f"CREATE OR REPLACE TABLE {self.table_name} AS SELECT * FROM sales_df")

    def query_df(self, sql: str):
        return self.conn.execute(sql).fetchdf()
