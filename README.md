# AFA MVP

Minimal Autonomous Financial Analyst MVP using FastAPI + Streamlit. Tools (MCP-like) server fetches data from Finnhub; backend orchestrates; frontend displays KPIs and news.

## Prerequisites
- Python 3.10+
- Finnhub API key (free tier)

## Setup
1. Create and activate a virtual environment, then install dependencies:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Create a `.env` file from `.env.example` and fill in values:
```
FINNHUB_KEY=YOUR_FINNHUB_KEY
OPENROUTER_API_KEY=  # optional for MVP
MCP_SERVER_URL=http://127.0.0.1:8081
BACKEND_URL=http://127.0.0.1:8000
```

## Run
1. Start the MCP tools server (Finnhub data + news):
```bash
python -m mcp_server.finance_mcp
```

2. Start the backend API:
```bash
python -m backend.main
```

3. Start the Streamlit frontend:
```bash
streamlit run frontend/app.py
```

Open the URL printed by Streamlit (usually http://localhost:8501). Use the sidebar to select a ticker and run analysis. If you see rate limit errors, wait and retry.

## Endpoints
- MCP server:
  - `GET /get_stock_time_series?ticker=AAPL&start=YYYY-MM-DD&end=YYYY-MM-DD`
  - `GET /fetch_news?ticker=AAPL&start=YYYY-MM-DD&end=YYYY-MM-DD&limit=10`
- Backend:
  - `GET /analyze?ticker=AAPL&lookback_days=365`

## Notes
- Educational use only. Not financial advice.
- LangChain and vector DB are included for future phases but not required for MVP.
