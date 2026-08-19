#!/usr/bin/env python3
"""
ELITE AI TRADING BOT  –  v2 (Institutional Upgrade)
===================================================
Highest-conviction, risk-first system used by serious independent traders.

Upgrades in this version:
✓ ATR-based trailing stops (dynamic, volatility-aware)
✓ Volatility regime filter (stay in cash when markets are chaotic)
✓ Multi-asset portfolio support
✓ Strict position sizing + leverage control
✓ Full transaction cost + slippage modeling
✓ Clean equity curve + trade log

Philosophy: Only take high-probability trades. Protect capital first.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from lightgbm import LGBMClassifier
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings("ignore")


class EliteTradingBot:
    def __init__(
        self,
        tickers: List[str] = ["SPY"],
        start: str = "2018-01-01",
        end: Optional[str] = None,
        capital: float = 100_000.0,
        risk_per_trade: float = 0.0075,          # 0.75% risk per trade
        max_leverage: float = 1.5,
        commission: float = 0.0005,              # 5 bps round trip
        slippage: float = 0.0002,
        long_threshold: float = 0.58,
        short_threshold: float = 0.42,
        atr_period: int = 14,
        atr_stop_mult: float = 2.2,              # Initial stop
        atr_trail_mult: float = 2.8,             # Trailing stop
        max_hold_days: int = 8,
        vol_lookback: int = 63,
        max_vol_percentile: float = 0.85,        # Regime filter
    ):
        self.tickers = [t.upper() for t in tickers]
        self.start = start
        self.end = end
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.max_leverage = max_leverage
        self.commission = commission
        self.slippage = slippage
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.atr_trail_mult = atr_trail_mult
        self.max_hold_days = max_hold_days
        self.vol_lookback = vol_lookback
        self.max_vol_percentile = max_vol_percentile

        self.models: Dict[str, LGBMClassifier] = {}
        self.features = None
        self.data: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Data & Features
    # ------------------------------------------------------------------
    def _download(self, ticker: str) -> pd.DataFrame:
        df = yf.download(ticker, start=self.start, end=self.end, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["ret_1"] = df["Close"].pct_change()
        df["ret_5"] = df["Close"].pct_change(5)
        df["ret_21"] = df["Close"].pct_change(21)
        df["vol_21"] = df["ret_1"].rolling(21).std()
        df["vol_63"] = df["ret_1"].rolling(63).std()
        df["vol_ratio"] = df["vol_21"] / df["vol_63"]

        df["rsi_14"] = self._rsi(df["Close"], 14)
        df["rsi_28"] = self._rsi(df["Close"], 28)
        df["mom_10"] = df["Close"] / df["Close"].shift(10) - 1
        df["mom_21"] = df["Close"] / df["Close"].shift(21) - 1

        df["vol_ma_20"] = df["Volume"].rolling(20).mean()
        df["rel_volume"] = df["Volume"] / df["vol_ma_20"]

        df["body"] = (df["Close"] - df["Open"]) / df["Open"]
        df["range"] = (df["High"] - df["Low"]) / df["Open"]
        df["upper_wick"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / df["Open"]
        df["lower_wick"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / df["Open"]

        # ATR for stops
        df["atr"] = self._atr(df, self.atr_period)

        # Regime: rolling vol percentile
        df["vol_percentile"] = df["vol_21"].rolling(self.vol_lookback).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        df = df.dropna()

        if self.features is None:
            self.features = [
                c for c in df.columns
                if c not in ["Open", "High", "Low", "Close", "Volume", "target", "atr", "vol_percentile"]
            ]
        return df

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    def train(self, ticker: str, train_df: pd.DataFrame):
        X = train_df[self.features]
        y = train_df["target"]

        model = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.025,
            max_depth=5,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=40,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X, y)
        self.models[ticker] = model
        return model

    def generate_signals(self, ticker: str, df: pd.DataFrame) -> pd.DataFrame:
        model = self.models[ticker]
        probs = model.predict_proba(df[self.features])[:, 1]
        df = df.copy()
        df["prob_up"] = probs
        df["signal"] = 0

        # High conviction + regime filter
        long_mask = (df["prob_up"] >= self.long_threshold) & (df["vol_percentile"] <= self.max_vol_percentile)
        short_mask = (df["prob_up"] <= self.short_threshold) & (df["vol_percentile"] <= self.max_vol_percentile)

        df.loc[long_mask, "signal"] = 1
        df.loc[short_mask, "signal"] = -1
        return df

    # ------------------------------------------------------------------
    # Single-asset backtest with ATR trailing stops
    # ------------------------------------------------------------------
    def backtest_ticker(self, ticker: str, df: pd.DataFrame, capital_slice: float):
        equity = capital_slice
        position = 0
        entry_price = 0.0
        stop_price = 0.0
        entry_idx = 0
        shares = 0.0
        trades = []
        equity_curve = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i - 1]

            # ----- Manage open position -----
            if position != 0:
                hold_days = i - entry_idx

                # Update trailing stop
                if position == 1:  # Long
                    new_trail = row["Close"] - self.atr_trail_mult * row["atr"]
                    stop_price = max(stop_price, new_trail)
                    hit_stop = row["Low"] <= stop_price
                else:  # Short
                    new_trail = row["Close"] + self.atr_trail_mult * row["atr"]
                    stop_price = min(stop_price, new_trail)
                    hit_stop = row["High"] >= stop_price

                exit_signal = (
                    hit_stop
                    or row["signal"] == -position
                    or hold_days >= self.max_hold_days
                )

                if exit_signal:
                    # Realistic exit
                    if hit_stop:
                        exit_price = stop_price * (1 - self.slippage * np.sign(position))
                    else:
                        exit_price = row["Open"] * (1 - self.slippage * np.sign(position))

                    gross_ret = position * (exit_price - entry_price) / entry_price
                    net_ret = gross_ret - self.commission * 2
                    notional = shares * entry_price
                    pnl = notional * net_ret
                    equity += pnl

                    trades.append({
                        "ticker": ticker,
                        "entry_date": df.index[entry_idx],
                        "exit_date": df.index[i],
                        "direction": "LONG" if position == 1 else "SHORT",
                        "entry_price": round(entry_price, 4),
                        "exit_price": round(exit_price, 4),
                        "shares": round(shares, 2),
                        "net_return": round(net_ret, 5),
                        "pnl": round(pnl, 2),
                        "hold_days": hold_days,
                        "exit_reason": "STOP" if hit_stop else ("SIGNAL" if row["signal"] == -position else "TIME"),
                    })
                    position = 0
                    shares = 0.0

            # ----- New entry -----
            if position == 0 and row["signal"] != 0:
                atr = prev["atr"]
                if pd.isna(atr) or atr <= 0:
                    continue

                risk_per_share = atr * self.atr_stop_mult
                risk_amount = equity * self.risk_per_trade
                shares = risk_amount / risk_per_share

                # Leverage cap
                notional = shares * row["Open"]
                max_notional = equity * self.max_leverage
                if notional > max_notional:
                    shares = max_notional / row["Open"]

                position = int(row["signal"])
                entry_price = row["Open"] * (1 + self.slippage * position)
                entry_idx = i
                shares = abs(shares)

                # Initial stop
                if position == 1:
                    stop_price = entry_price - self.atr_stop_mult * atr
                else:
                    stop_price = entry_price + self.atr_stop_mult * atr

            equity_curve.append({"date": df.index[i], "equity": equity, "ticker": ticker})

        # Force close at end
        if position != 0:
            last = df.iloc[-1]
            exit_price = last["Close"] * (1 - self.slippage * np.sign(position))
            gross_ret = position * (exit_price - entry_price) / entry_price
            net_ret = gross_ret - self.commission * 2
            pnl = shares * entry_price * net_ret
            equity += pnl
            trades.append({
                "ticker": ticker,
                "entry_date": df.index[entry_idx],
                "exit_date": df.index[-1],
                "direction": "LONG" if position == 1 else "SHORT",
                "entry_price": round(entry_price, 4),
                "exit_price": round(exit_price, 4),
                "shares": round(shares, 2),
                "net_return": round(net_ret, 5),
                "pnl": round(pnl, 2),
                "hold_days": len(df) - 1 - entry_idx,
                "exit_reason": "EOD",
            })

        return pd.DataFrame(trades), pd.DataFrame(equity_curve), equity

    # ------------------------------------------------------------------
    # Portfolio runner
    # ------------------------------------------------------------------
    def run(self, train_ratio: float = 0.70):
        print("=" * 60)
        print("  ELITE AI TRADING BOT  v2  –  Multi-Asset + ATR Trailing")
        print("=" * 60)

        all_trades = []
        all_equity = []
        final_equities = {}

        capital_per_ticker = self.capital / len(self.tickers)

        for ticker in self.tickers:
            print(f"\n[*] Processing {ticker}...")
            raw = self._download(ticker)
            if len(raw) < 300:
                print(f"    Skipping {ticker} – insufficient data")
                continue

            df = self.engineer_features(raw)
            split = int(len(df) * train_ratio)
            train_df = df.iloc[:split]
            test_df = df.iloc[split:].copy()

            self.train(ticker, train_df)
            test_df = self.generate_signals(ticker, test_df)

            trades, equity_curve, final_eq = self.backtest_ticker(ticker, test_df, capital_per_ticker)
            all_trades.append(trades)
            all_equity.append(equity_curve)
            final_equities[ticker] = final_eq

            print(f"    Final equity for {ticker}: ${final_eq:,.2f}")

        # Aggregate
        if not all_trades:
            print("\n[!] No trades generated across the portfolio.")
            return None, None

        trades_df = pd.concat(all_trades, ignore_index=True)
        equity_df = pd.concat(all_equity, ignore_index=True)

        # Portfolio equity (simple sum of per-ticker equity curves aligned by date)
        portfolio_equity = equity_df.groupby("date")["equity"].sum().sort_index()
        total_final = portfolio_equity.iloc[-1] if len(portfolio_equity) else self.capital
        total_return = (total_final / self.capital - 1) * 100

        # Stats
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        win_rate = len(wins) / len(trades_df) if len(trades_df) else 0
        expectancy = trades_df["pnl"].mean() if len(trades_df) else 0

        print("\n" + "=" * 60)
        print("               PORTFOLIO RESULTS")
        print("=" * 60)
        print(f"Tickers              : {', '.join(self.tickers)}")
        print(f"Starting Capital     : ${self.capital:,.0f}")
        print(f"Final Portfolio Equity: ${total_final:,.2f}")
        print(f"Total Return         : {total_return:+.2f}%")
        print(f"Total Trades         : {len(trades_df)}")
        print(f"Win Rate             : {win_rate:.1%}")
        print(f"Expectancy / trade   : ${expectancy:,.2f}")
        print("=" * 60)

        # Save
        trades_df.to_csv("trades.csv", index=False)
        portfolio_equity.to_csv("equity_curve.csv")
        print("\n[+] Results saved → trades.csv & equity_curve.csv")

        return trades_df, portfolio_equity


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example: Single name (fast) or portfolio
    bot = EliteTradingBot(
        tickers=["SPY"],                     # ← Change to ["SPY", "QQQ", "IWM"] for portfolio
        start="2018-01-01",
        capital=100_000,
        risk_per_trade=0.0075,
        long_threshold=0.58,
        short_threshold=0.42,
        atr_stop_mult=2.2,
        atr_trail_mult=2.8,
        max_hold_days=8,
        max_vol_percentile=0.85,             # Stay out of top 15% volatility regimes
    )

    trades, equity = bot.run()
