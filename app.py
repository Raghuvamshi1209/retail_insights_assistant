import os
import streamlit as st
import pandas as pd

from retail_assistant.utils import load_csv, make_duckdb_conn, get_schema_metadata
from retail_assistant.graph import build_chat_graph, build_summary_graph

st.set_page_config(page_title="Retail Insights Assistant", layout="wide")
st.title("🛍️ Retail Insights Assistant")

with st.sidebar:
    st.header("🔑 Setup")
    st.code('setx GEMINI_API_KEY "YOUR_KEY"', language="bat")
    st.caption("Restart terminal after setx.")

    st.markdown("---")
    st.header("⚙️ Settings")
    gemini_model = st.text_input("Gemini model", value=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    temperature = st.slider("Temperature", 0.0, 1.0, float(os.getenv("TEMPERATURE", "0.1")), 0.05)
    max_rows = st.number_input("Max rows to display", 5, 500, int(os.getenv("MAX_ROWS", "5")), 5)

    st.markdown("---")
    if st.button("♻️ Reset session / reload graphs"):
        st.session_state.clear()
        st.rerun()

if not os.getenv("GEMINI_API_KEY"):
    st.warning("GEMINI_API_KEY is not set. Set it (see sidebar) and restart the app.")

uploaded = st.file_uploader("Upload Sales CSV", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV to start")
    st.stop()

df = load_csv(uploaded)

for col in ["Amount", "Qty"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

conn = make_duckdb_conn()
conn.register("sales_df", df)
conn.execute("CREATE OR REPLACE TABLE sales AS SELECT * FROM sales_df")

schema_md = get_schema_metadata(df)

st.subheader("Dataset Preview")
st.write(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
st.dataframe(df.head(int(max_rows)), use_container_width=True)

settings_key = (gemini_model, float(temperature))
if "_settings_key" not in st.session_state or st.session_state.get("_settings_key") != settings_key:
    st.session_state["_settings_key"] = settings_key
    st.session_state["chat_graph"] = build_chat_graph(gemini_model=gemini_model, temperature=float(temperature))
    st.session_state["summary_graph"] = build_summary_graph(gemini_model=gemini_model, temperature=float(temperature))

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

tab1, tab2 = st.tabs(["📌 Summarize", "💬 Chat Q&A"])

with tab1:
    st.markdown("Deterministic KPIs + Gemini narrative summary.")
    if st.button("Generate Summary", type="primary"):
        with st.spinner("Generating summary..."):
            state = {
                "schema": schema_md,
                "chat_history": st.session_state.chat_history,
                "max_rows": int(max_rows),
                "duckdb_conn": conn,
            }
            result = st.session_state.summary_graph.invoke(state)

        st.success("Summary ready")
        st.markdown(result.get("answer", "(no answer)"))
        if result.get("warnings"):
            st.warning("\n".join(result["warnings"]))
        tables = result.get("_summary_tables")
        if isinstance(tables, dict) and len(tables)>0:
            with st.expander("Show supporting tables (KPI's / Trends / Top Lists)"):
                for name, tdf in tables.items():
                    st.markdown(f'### {name}')
                    try:
                        st.dataframe(tdf.head(int(max_rows)), use_container_width=True)
                    except:
                        st.write(tdf)

with tab2:
    st.markdown("Ask questions about the dataset.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    q = st.chat_input("Ask a question")
    if q:
        st.session_state.chat_history.append({"role": "user", "content": q})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                state = {
                    "user_query": q,
                    "schema": schema_md,
                    "chat_history": st.session_state.chat_history[-10:],
                    "max_rows": int(max_rows),
                    "duckdb_conn": conn,
                }
                result = st.session_state.chat_graph.invoke(state)

            st.markdown(result.get("answer", "(no answer)"))

            if result.get("warnings"):
                st.warning("\n".join(result["warnings"]))

            with st.expander("Show SQL"):
                st.code(result.get("sql", ""), language="sql")

            if isinstance(result.get("result_df"), pd.DataFrame) and len(result["result_df"]) > 0:
                st.dataframe(result["result_df"].head(int(max_rows)), use_container_width=True)

        st.session_state.chat_history.append({"role": "assistant", "content": result.get("answer", "")})
