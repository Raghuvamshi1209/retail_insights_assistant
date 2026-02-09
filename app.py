from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Ensure src is importable when running `streamlit run app.py`
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from retail_ai.data_engine.data_loader import DataLoader
from retail_ai.engine import RetailAssistantEngine
from retail_ai.handlers.error_handler import friendly_error


def main() -> None:
    load_dotenv()

    st.set_page_config(page_title="Retail Insights Assistant", layout="wide")
    st.title("Retail Insights Assistant")

    input_dir = Path(os.getenv("DATA_INPUT_DIR", "data/input"))
    loader = DataLoader(input_dir=input_dir)

    with st.sidebar:
        st.header("Dataset")
        st.caption(f"Place CSV files in: {input_dir}")
        if st.button("Refresh file list"):
            st.rerun()

        max_rows = st.number_input(
            "Max rows to display", 10, 500, int(os.getenv("MAX_ROWS", "50")), 10
        )

    files = loader.list_csv_files()
    if not files:
        st.warning(
            "No CSV files found in data/input. Copy your dataset CSV into data/input/ and refresh."
        )
        st.stop()

    selected = st.selectbox("Select a CSV file", options=files, format_func=lambda p: p.name)

    try:
        df = loader.load(Path(selected))
    except Exception as e:
        st.error(friendly_error(e))
        st.stop()

    st.subheader("Dataset Preview")
    st.write(f"File: **{Path(selected).name}** | Rows: {len(df):,} | Columns: {df.shape[1]}")
    st.dataframe(df.head(int(max_rows)), use_container_width=True)

    # Engine cached per session
    if "engine" not in st.session_state:
        st.session_state.engine = RetailAssistantEngine()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    tab1, tab2 = st.tabs(["Summarize", "Chat Q&A"])

    with tab1:
        if st.button("Generate Summary", type="primary"):
            try:
                res = st.session_state.engine.summarize(df, max_rows=int(max_rows))
                st.success("Summary ready")
                st.markdown(res.get("answer", "(no answer)"))

                tables = res.get("_summary_tables") or {}
                if tables:
                    with st.expander("Show supporting tables"):
                        for name, tdf in tables.items():
                            st.markdown(f"### {name}")
                            st.dataframe(tdf.head(int(max_rows)), use_container_width=True)
            except Exception as e:
                st.error(friendly_error(e))

    with tab2:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        q = st.chat_input("Ask a question")
        if q:
            st.session_state.chat_history.append({"role": "user", "content": q})
            with st.chat_message("assistant"):
                try:
                    res = st.session_state.engine.answer(
                        df,
                        q,
                        chat_history=st.session_state.chat_history[-10:],
                        max_rows=int(max_rows),
                    )
                    st.markdown(res.get("answer", "(no answer)"))

                    with st.expander("Show SQL"):
                        st.code(res.get("sql", ""), language="sql")

                    if res.get("result_df") is not None and len(res["result_df"]) > 0:
                        st.dataframe(res["result_df"].head(int(max_rows)), use_container_width=True)
                except Exception as e:
                    st.error(friendly_error(e))
            st.session_state.chat_history.append(
                {"role": "assistant", "content": res.get("answer", "")}
            )


if __name__ == "__main__":
    main()
