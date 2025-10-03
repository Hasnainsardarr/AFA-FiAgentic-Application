import os
import requests
import pandas as pd
from dotenv import load_dotenv
import streamlit as st


load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")


st.set_page_config(page_title="AFA MVP", layout="wide")
st.title("Autonomous Financial Analyst - MVP")
st.caption("For educational purposes only. Not financial advice.")

with st.sidebar:
	st.header("Settings")
	ticker = st.text_input("Ticker", value="AAPL").upper().strip()
	lookback_days = st.slider("Lookback (days)", min_value=1, max_value=60, value=7, step=1)
	analyze = st.button("Run Analysis")

col_left, col_right = st.columns([3, 2])

if analyze and ticker:
	with st.spinner("Analyzing..."):
		try:
			resp = requests.get(f"{BACKEND_URL}/analyze", params={"ticker": ticker, "lookback_days": lookback_days}, timeout=60)
			resp.raise_for_status()
			result = resp.json()
			kpis = result.get("kpis", {})
			news = result.get("news", [])
			st.success("Analysis complete")

			with col_left:
				st.subheader("KPIs")
				st.dataframe(pd.DataFrame([kpis]))

			with col_right:
				st.subheader("Recent News")
				for item in news:
					headline = item.get("headline") or "(no headline)"
					url = (item.get("url") or "").strip()
					source = item.get("source") or ""
					if url.startswith("http://") or url.startswith("https://"):
						st.markdown(f"- [{headline}]({url}) — {source}")
					else:
						st.markdown(f"- {headline} — {source}")

		except requests.HTTPError as e:
			st.error(f"Backend error: {e.response.text if e.response else e}")
		except Exception as e:  # noqa: BLE001
			st.error(str(e))

st.divider()
st.write("Need help? Ensure MCP server and backend are running, and .env is set.")


