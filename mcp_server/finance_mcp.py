import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Finance MCP Server (MVP)")

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


@app.get("/get_stock_time_series")
def get_stock_time_series(
    ticker: str = Query(..., min_length=1, max_length=12),
    interval: str = Query("1d", pattern="^(1d|1h|30m|15m|5m|1m)$"),
    start: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end: Optional[str] = Query(None, description="YYYY-MM-DD"),
) -> Dict[str, Any]:
    """
    Fetch OHLCV data from Yahoo Finance.
    If start/end are omitted, defaults to last 2 days for daily data
    or last 24 hours for intraday data.
    """
    if end:
        end_dt = datetime.strptime(end, "%Y-%m-%d")
    else:
        end_dt = datetime.now()

    if start:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
    else:
        # Default to small range based on interval
        if interval == "1d":
            start_dt = end_dt - timedelta(days=2)
        else:
            start_dt = end_dt - timedelta(hours=24)

    # Get stock data from yfinance
    stock = yf.Ticker(ticker.upper())
    df = stock.history(
        start=start_dt,
        end=end_dt,
        interval=interval,
        prepost=False  # Regular market hours only
    )

    # Convert to list of dictionaries
    series = []
    for index, row in df.iterrows():
        series.append({
            "date": index.strftime("%Y-%m-%d %H:%M:%S"),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": int(row["Volume"])
        })

    return {
        "ticker": ticker.upper(),
        "interval": interval,
        "start": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "end": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "series": series,
    }


@app.get("/fetch_news")
def fetch_news(
    ticker: str = Query(..., min_length=1, max_length=12),
    limit: int = Query(8, ge=1, le=50),
) -> Dict[str, Any]:
    """Fetch latest company news from Yahoo Finance (no historical filter)."""
    stock = yf.Ticker(ticker.upper())
    items = stock.news or []
    items = items[:limit]

    news = []
    for it in items:
        published_ts = it.get("providerPublishTime")
        news.append({
            "title": it.get("title"),
            "link": it.get("link"),
            "publisher": it.get("publisher"),
            "published_at": (
                datetime.fromtimestamp(published_ts).isoformat() if published_ts else None
            ),
            "summary": it.get("summary"),
        })

    return {
        "ticker": ticker.upper(),
        "news": news,
    }



@app.get("/analyze")
def analyze(
    ticker: str = Query(..., min_length=1, max_length=12),
    interval: str = Query("1d", pattern="^(1d|1h|30m|15m|5m|1m)$"),
    lookback_days: int = Query(2, ge=1, le=365)
) -> Dict[str, Any]:
    """
    Analyze a stock: fetch OHLCV data and news.
    For intraday intervals, lookback_days is converted to hours.
    """
    # 1. Fetch stock prices
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=lookback_days)
    stock_data = get_stock_time_series(
        ticker=ticker,
        interval=interval,
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d")
    )

    # 2. Fetch news
    news = fetch_news(ticker=ticker, limit=5)["news"]

    return {
        "ticker": ticker,
        "interval": interval,
        "lookback_days": lookback_days,
        "data": stock_data["series"],
        "news": news
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
