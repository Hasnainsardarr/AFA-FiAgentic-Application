import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
from backend.analysis import compute_kpis
from backend.agent_runner import run_agent_query

load_dotenv()


MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8081").rstrip("/")

app = FastAPI(title="AFA Backend (MVP)")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
	return {"status": "ok"}
class AgentQuery(BaseModel):
    input: str


@app.post("/agent_query")
def agent_query(body: AgentQuery) -> Dict[str, Any]:
    """Run a free-form query through the agent and return the response text."""
    if not body.input or not body.input.strip():
        raise HTTPException(status_code=400, detail="'input' is required")
    result = run_agent_query(body.input.strip())
    return {"result": result}


@app.get("/agent_query")
def agent_query_get(q: str = Query(..., min_length=1)) -> Dict[str, Any]:
    """Convenience GET variant for quick testing in a browser or via curl."""
    result = run_agent_query(q.strip())
    return {"result": result}


def _date_str(dt: datetime) -> str:
	return dt.strftime("%Y-%m-%d")


def _fetch_series(ticker: str, start: Optional[str], end: Optional[str]) -> List[Dict[str, Any]]:
	params = {"ticker": ticker}
	if start:
		params["start"] = start
	if end:
		params["end"] = end
	resp = requests.get(f"{MCP_SERVER_URL}/get_stock_time_series", params=params, timeout=30)
	if resp.status_code != 200:
		raise HTTPException(status_code=resp.status_code, detail=resp.text)
	return resp.json().get("series", [])


def _fetch_news(ticker: str, start: Optional[str], end: Optional[str], limit: int = 10) -> List[Dict[str, Any]]:
	params = {"ticker": ticker, "limit": limit}
	if start:
		params["start"] = start
	if end:
		params["end"] = end
	resp = requests.get(f"{MCP_SERVER_URL}/fetch_news", params=params, timeout=30)
	if resp.status_code != 200:
		raise HTTPException(status_code=resp.status_code, detail=resp.text)
	return resp.json().get("news", [])


@app.get("/analyze")
def analyze(
	ticker: str = Query(..., min_length=1, max_length=12, description="Ticker symbol, e.g. AAPL"),
	lookback_days: int = Query(7, ge=1, le=1825),
) -> Dict[str, Any]:
	"""Run the MVP analysis: fetch series + news via MCP tools and compute KPIs."""
	end_dt = datetime.utcnow()
	start_dt = end_dt - timedelta(days=lookback_days)
	start_str, end_str = _date_str(start_dt), _date_str(end_dt)

	series = _fetch_series(ticker, start_str, end_str)
	kpis = compute_kpis(series)
	news = _fetch_news(ticker, _date_str(end_dt - timedelta(days=7)), end_str, limit=8)

	return {
		"ticker": ticker.upper(),
		"period": {"start": start_str, "end": end_str},
		"kpis": kpis,
		"time_series_points": len(series),
		"news_count": len(news),
		"news": news,
		"disclaimer": "For educational purposes only. Not financial advice.",
	}


if __name__ == "__main__":
	import uvicorn
	uvicorn.run(app, host="0.0.0.0", port=8000)


