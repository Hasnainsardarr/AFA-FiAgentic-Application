Project Overview
This project is to build an Autonomous Financial Analyst Agent (AFA) that uses the Model Context Protocol (MCP) for tool integration. The AFA fetches live market data and news, computes financial KPIs (e.g., volatility, Sharpe ratio, growth, basic backtesting), cross-checks data for confidence, stores findings in vector memory, and generates reports and dashboards.
The full product includes advanced features like critic agents, long-term memory, PDF/CSV exports, and security hardening. However, focus on the initial MVP first. Build incrementally: start with core tools, basic analysis, and a simple UI. Later phases will add critic verification, memory, reports, and deployment.
Key assumption: Using Python (FastAPI for backend, LangChain for agent orchestration, Pinecone for vector memory, Streamlit for UI). For LLMs, use OpenRouter free models (e.g., via their API, compatible with OpenAI SDK by setting base URL and API key). No strict dependencies on paid OpenAI—adapt code to OpenRouter.
MVP Scope (Phase 1 Focus)
The MVP is a minimal working product:

MCP Server: Expose basic tools for fetching stock time-series data and news (using Finnhub API free tier).
Backend Orchestrator: Use LangChain to wrap MCP tools, fetch data for a ticker, compute simple KPIs (e.g., annual return, volatility, Sharpe ratio), and return JSON results.
Frontend UI: A basic Streamlit app where users input a ticker, trigger analysis, and view results (e.g., KPIs in a table). No memory, critic, or reports yet.
No advanced features: Skip critic agent, vector memory, PDF exports, deployment, or security audits for MVP. Add them in later iterations.

Milestones for MVP:

Set up MCP server with get_stock_time_series and fetch_news tools.
Build backend to call these tools via MCP client and compute KPIs.
Create Streamlit UI to interact with backend and display results.

Once MVP is working, iterate: add critic (Phase 5), memory (Phase 4), reports (Phase 6), etc.
Tech Stack

Backend: Python 3.10+, FastAPI for API, LangChain for agents (use MCP adapters if available, or manual wrapping).
MCP: Use MCP Python SDK/FastMCP to expose tools.
Data Sources: Finnhub API (free tier for MVP; handle rate limits).
Analysis: Pandas, NumPy for KPI calculations.
LLM: OpenRouter free models (e.g., set base_url="https://openrouter.ai/api/v1" and use your API key). Use for agent reasoning if needed, but MVP can start with rule-based analysis.
UI: Streamlit for quick prototyping.
Vector DB: Pinecone (free tier)—but skip for MVP; add in later phase.
Environment: Use .env for keys (e.g., FINNHUB_KEY, OPENROUTER_API_KEY). No Docker yet.

Install core packages:
textpip install fastapi uvicorn requests pandas numpy langchain streamlit pinecone-client
(Adjust for exact MCP SDK package; assume mcp-server-sdk or similar.)

Folder Structure

/mcp_server: MCP tool implementations (e.g., finance_mcp.py).
/backend: Agent orchestration and FastAPI (e.g., agent_runner.py, analysis.py).
/frontend: Streamlit app (e.g., app.py).
/scripts: Helpers (e.g., report generation—add later).
/docs: README, diagrams.

Key Implementation Notes

MCP Integration: Expose tools with @mcp.tool() decorators. Run MCP server separately (e.g., on port 8081).
Agent: Use LangChain to initialize agent with MCP-wrapped tools. For MVP, keep it simple—focus on data fetch and KPI computation.
OpenRouter Adaptation: In code, replace from openai import OpenAI with OpenRouter-compatible client (e.g., openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)).
Rate Limits: Finnhub free tier has limits—add caching if needed, but keep MVP simple.
Security/Ethics: For MVP, add basic input validation. Later, add disclaimers ("Not financial advice") and MCP hardening.
Testing: Manually test MCP tools via curl, backend via Postman, UI via browser.