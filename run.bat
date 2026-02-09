@echo off
setlocal

REM Create venv if missing
if not exist .venv (
  echo Creating venv...
  python -m venv .venv
)

call .venv\Scripts\activate

echo Installing dependencies...
pip install -r requirements.txt

if not exist .env (
  echo Creating .env from .env.example...
  copy .env.example .env >nul
  echo Please edit .env and set GEMINI_API_KEY
)

echo Starting Streamlit...
streamlit run app.py

endlocal
