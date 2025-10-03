import os
import json
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

# LangChain core + OpenAI LLM
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from langchain.agents import AgentType, initialize_agent


# =============================
# Agent Runner for AFA (Phase 2)
# =============================
# This script wires a LangChain agent to your tool server ("MCP server") at
# http://localhost:8081. It discovers available endpoints, wraps them as
# LangChain Tools, and initializes a ZERO_SHOT_REACT_DESCRIPTION agent that can
# call these tools dynamically to accomplish multi-step tasks like:
# planner → fetch → analyze → critic → writer
#
# Usage:
#   1) Ensure your tool server is running: python mcp_server/finance_mcp.py
#   2) Set OPENROUTER_API_KEY in your environment or .env
#   3) Run this file: python backend/agent_runner.py
#
# Notes for beginners:
# - The "MCP server" here is your FastAPI tool service exposing endpoints like
#   /get_stock_time_series and /fetch_news. We "discover" these via its
#   OpenAPI spec at /openapi.json and dynamically create tools the agent can use.
# - The agent uses an LLM (ChatOpenAI) to decide which tool to call and with
#   what inputs based on the user question.


load_dotenv()


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


# ---------
# Inputs for tools (schemas the agent uses to validate parameters)
# ---------
class StockTimeSeriesInput(BaseModel):
    ticker: str = Field(..., description="Ticker symbol, e.g. AAPL")
    start: Optional[str] = Field(None, description="YYYY-MM-DD start date")
    end: Optional[str] = Field(None, description="YYYY-MM-DD end date")
    interval: str = Field(
        "1d",
        description="Candle interval: 1d, 1h, 30m, 15m, 5m, 1m",
        pattern=r"^(1d|1h|30m|15m|5m|1m)$",
    )


class FetchNewsInput(BaseModel):
    ticker: str = Field(..., description="Ticker symbol, e.g. AAPL")
    limit: int = Field(5, ge=1, le=50, description="Max number of articles")


# ---------
# Tool wrapper functions (call the HTTP endpoints on the tool server)
# ---------
def _http_get_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as e:
        text = e.response.text if e.response is not None else str(e)
        raise RuntimeError(f"HTTP error calling {url}: {text}") from e
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Request failed for {url}: {e}") from e


def call_get_stock_time_series(
    mcp_base_url: str, args: StockTimeSeriesInput
) -> str:
    """Fetch OHLCV from the tool server. Returns JSON string for transparency."""
    data = _http_get_json(
        f"{mcp_base_url}/get_stock_time_series",
        {
            "ticker": args.ticker,
            "start": args.start,
            "end": args.end,
            "interval": args.interval,
        },
    )
    # Return compact JSON so the agent can easily read/quote it
    return json.dumps(data, ensure_ascii=False)


def call_fetch_news(mcp_base_url: str, args: FetchNewsInput) -> str:
    """Fetch recent news from the tool server. Returns JSON string."""
    data = _http_get_json(
        f"{mcp_base_url}/fetch_news",
        {"ticker": args.ticker, "limit": args.limit},
    )
    return json.dumps(data, ensure_ascii=False)


# ---------
# Discovery: read the server's OpenAPI to see which tools exist
# ---------
def discover_endpoints(mcp_base_url: str) -> Dict[str, Any]:
    spec_url = f"{mcp_base_url}/openapi.json"
    try:
        spec = _http_get_json(spec_url, {})
        return spec.get("paths", {})
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Could not retrieve OpenAPI from {spec_url}. Is the server running? {e}"
        ) from e


def build_tools(mcp_base_url: str, paths: Dict[str, Any]) -> List[StructuredTool]:
    """Create LangChain Tools for any known endpoints found via discovery."""
    tools: List[StructuredTool] = []

    # Map known endpoints → tool wrappers (add more as your server grows)
    known = {
        "/get_stock_time_series": {
            "name": "get_stock_time_series",
            "desc": (
                "Fetch OHLCV candles via Yahoo Finance for a ticker and date window. "
                "Inputs: ticker (str), start (YYYY-MM-DD), end (YYYY-MM-DD), interval (1d|1h|30m|15m|5m|1m). "
                "Returns JSON with series of {date, open, high, low, close, volume}."
            ),
            "schema": StockTimeSeriesInput,
            "func_factory": lambda: (  # bind base URL into the closure
                lambda args: call_get_stock_time_series(mcp_base_url, args)
            ),
        },
        "/fetch_news": {
            "name": "fetch_news",
            "desc": (
                "Fetch recent company news via Yahoo Finance for a ticker. "
                "Inputs: ticker (str), limit (int, 1-50). Returns JSON with articles."
            ),
            "schema": FetchNewsInput,
            "func_factory": lambda: (
                lambda args: call_fetch_news(mcp_base_url, args)
            ),
        },
        # Example placeholder for future server feature:
        "/save_report": {
            "name": "save_report",
            "desc": (
                "Save a summarized analysis/report. If unavailable, the agent should fall back to returning text."
            ),
            "schema": None,  # Will be added when server supports it
            "func_factory": None,
        },
    }

    for path, meta in known.items():
        if path not in paths:
            # Graceful handling: tool not available, skip but keep the user informed later
            continue

        if meta["schema"] is None or meta["func_factory"] is None:
            # Discovered but not implemented locally yet
            continue

        tool = StructuredTool.from_function(
            name=meta["name"],
            description=meta["desc"],
            args_schema=meta["schema"],
            func=meta["func_factory"](),
        )
        tools.append(tool)

    return tools


def build_agent(tools: List[StructuredTool]) -> Any:
    """Create a ZERO_SHOT_REACT_DESCRIPTION agent with our tools using OpenRouter Mistral."""
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not openrouter_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Please set it in your environment or .env."
        )

    # Use OpenRouter's OpenAI-compatible API with Mistral 7B Instruct
    llm = ChatOpenAI(
        model="mistralai/mistral-7b-instruct",
        temperature=0,
        api_key=openrouter_key,
        base_url="https://openrouter.ai/api/v1",
    )

    # Initialize classic ReAct-style agent that calls tools by name in multi-step fashion
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )
    return agent


def run_demo(agent: Any, available_tools: List[StructuredTool]) -> str:
    """Run the example request end-to-end and return the final answer string."""
    tool_names = ", ".join(t.name for t in available_tools) or "(no tools discovered)"
    system_hint = (
        "You are an expert financial analysis assistant. Always plan your steps first, then act. "
        "Use a multi-step process: planner → fetch → analyze → critic → writer. "
        f"Available tools: {tool_names}. Call tools as needed to fetch data/news before concluding."
    )
    user_query = "Compare NVDA vs AMD, analyze risks, and give a recommendation."

    # The ReAct agent will include its own chain-of-thought. We simply prepend guidance.
    prompt = f"{system_hint}\n\nTask: {user_query}"

    try:
        result = agent.run(prompt)
        return result if isinstance(result, str) else str(result)
    except Exception as e:  # noqa: BLE001
        # Provide a helpful, compact error for beginners
        missing = []
        if not available_tools:
            missing.append("No tools discovered. Is the tool server running?")
        return (
            "Agent failed to complete the task.\n"
            + ("\n".join(missing) + "\n" if missing else "")
            + f"Error: {e}"
        )


def main() -> None:
    mcp_url = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8081").rstrip("/")

    # 1) Discover available endpoints from the server
    paths = discover_endpoints(mcp_url)

    # 2) Build LangChain Tools for discovered endpoints (gracefully skip unavailable ones)
    tools = build_tools(mcp_url, paths)

    # 3) Build the agent that can call those tools dynamically
    agent = build_agent(tools)

    # 4) Run the example multi-step query and print the final answer
    print("\n=== AFA Agent Demo ===")
    print(f"Tool server: {mcp_url}")
    print(f"Discovered tools: {[t.name for t in tools]}")
    print("Running demo query...\n")

    result = run_demo(agent, tools)
    print("\n--- Agent Final Answer ---\n")
    print(result)


if __name__ == "__main__":
    main()


# ==============================================================
# Helper API for backend imports: cached agent and query function
# ==============================================================
_CACHED_AGENT: Optional[Any] = None
_CACHED_TOOLS: List[StructuredTool] = []


def get_or_create_agent() -> Any:
    """Return a cached agent instance; create it on first use."""
    global _CACHED_AGENT, _CACHED_TOOLS
    if _CACHED_AGENT is not None:
        return _CACHED_AGENT

    mcp_url = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8081").rstrip("/")
    paths = discover_endpoints(mcp_url)
    _CACHED_TOOLS = build_tools(mcp_url, paths)
    _CACHED_AGENT = build_agent(_CACHED_TOOLS)
    return _CACHED_AGENT


def run_agent_query(user_input: str) -> str:
    """Run an arbitrary user query through the agent and return the final answer."""
    agent = get_or_create_agent()
    tool_names = ", ".join(t.name for t in _CACHED_TOOLS) or "(no tools discovered)"
    system_hint = (
        "You are an expert financial analysis assistant. Always plan your steps first, then act. "
        "Use a multi-step process: planner → fetch → analyze → critic → writer. "
        f"Available tools: {tool_names}. Call tools as needed to fetch data/news before concluding."
    )
    prompt = f"{system_hint}\n\nTask: {user_input}"

    try:
        result = agent.run(prompt)
        return result if isinstance(result, str) else str(result)
    except Exception as e:  # noqa: BLE001
        missing = []
        if not _CACHED_TOOLS:
            missing.append("No tools discovered. Is the tool server running?")
        return (
            "Agent failed to complete the task.\n"
            + ("\n".join(missing) + "\n" if missing else "")
            + f"Error: {e}"
        )


