# Retail Insights Assistant
# README / Technical Notes

## Setup and execution (Windows)

### 1) Install
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Configure Gemini API key
```bat
setx GEMINI_API_KEY "<your_key>"
```
Restart terminal after `setx`.

### 3) Run
```bat
streamlit run app.py
```

## Architecture summary
- **Streamlit** UI for file upload and chat.
- **LangGraph agents**: Planner → Validator → Extractor → Narrator.
- **DuckDB** executes SQL for deterministic KPIs and Q&A.
- **Gemini** is used for: (1) JSON planning, (2) narration.


## How it works (high-level)

### Q&A Mode
1. **Planner (Gemini)** produces a strict JSON plan (metrics, group_by, filters, time, sort, limit)
2. **Validator (rules)** removes unknown columns/metrics and enforces safe constraints
3. **Extractor (DuckDB)** generates **SELECT-only** SQL and executes it
4. **Narrator (Gemini)** converts results into a concise business answer

### Summarization Mode
- Runs deterministic KPI and trend SQL queries
- Gemini writes an executive summary using only computed stats


## Assumptions
- Amount = sales value; Qty = units
- Shipped revenue: Status starts with 'Shipped'
- Cancelled rows: Status contains 'Cancelled'

## Limitations
- Dataset date range: 2022-03-31 to 2022-06-29 (YoY not possible without multiple years)
- Gemini free tier limits may apply.

## Possible improvements
- Add driver analysis agent for “why” questions (MoM decomposition).
- Add evaluation harness (golden Q&A set → expected SQL).
- 100GB+: Parquet + partitions + Spark/Trino.

## Repo structure

```
codebase_retail_insights_assistant/
  app.py
  requirements.txt
  Amazon Sale Report.csv
  retail_assistant/
    graph.py
    prompts.py
    sql_builder.py
    validators.py
    utils.py
    llm_providers/
      gemini_chat.py
```
