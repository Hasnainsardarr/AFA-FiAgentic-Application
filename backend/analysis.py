from typing import Dict, List

import numpy as np
import pandas as pd


def compute_kpis(prices: List[Dict[str, float]], risk_free_rate: float = 0.0) -> Dict[str, float]:
	"""Compute simple KPIs from a list of {date, close}.

	Returns annual_return, volatility, sharpe_ratio and num_days.
	"""
	df = pd.DataFrame(prices)
	if df.empty or "close" not in df:
		return {
			"annual_return": None,
			"volatility": None,
			"sharpe_ratio": None,
			"num_days": 0,
		}

	df = df.sort_values("date")
	df["return"] = df["close"].pct_change()
	daily_returns = df["return"].dropna()
	if daily_returns.empty or float(daily_returns.std()) == 0.0:
		return {
			"annual_return": None,
			"volatility": None,
			"sharpe_ratio": None,
			"num_days": int(df.shape[0]),
		}

	mean_daily = float(daily_returns.mean())
	std_daily = float(daily_returns.std())
	trading_days = 252
	annual_return = (1.0 + mean_daily) ** trading_days - 1.0
	annual_volatility = std_daily * float(np.sqrt(trading_days))
	# Convert annual risk-free rate to daily for Sharpe calculation
	daily_rf = (1.0 + risk_free_rate) ** (1.0 / trading_days) - 1.0
	sharpe = float(np.sqrt(trading_days)) * ((mean_daily - daily_rf) / std_daily)

	return {
		"annual_return": float(annual_return),
		"volatility": float(annual_volatility),
		"sharpe_ratio": float(sharpe),
		"num_days": int(df.shape[0]),
	}


