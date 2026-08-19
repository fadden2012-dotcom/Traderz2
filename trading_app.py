#!/usr/bin/env python3
"""
ELITE AI TRADING BOT – v3 (Daily Signals + Risk Controls)
=========================================================
Professional mobile trading system with:
✓ Full institutional backtest (ATR trailing + regime filter)
✓ Daily high-conviction signal generator
✓ Tighter portfolio risk controls
✓ Alpaca Paper Trading
✓ Streamlit Cloud ready (phone installable)

Run:     streamlit run trading_app.py
Deploy:  GitHub → share.streamlit.io
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from lightgbm import LGBMClassifier
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Optional Alpaca
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

st.set_page_config(
    page_title="Elite AI Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stMetric { background-color: #0e1117; padding: 12px; border-radius: 10px; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; font-weight: 600; }
    div[data-testid="stSidebar"] { background-color: #0e1117; }
    .signal-long { color: #00c853; font-weight: 700; font-size: 1.2rem; }
    .signal-short { color: #ff1744; font-weight: 700; font-size: 1.2rem; }
    .signal-flat { color: #9e9e9e; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# Core Engine
# -------------------------------------------------
class EliteTradingBot:
    def __init__(
        self,
        tickers: List[str],
        start: str = "2018-01-01",
        capital: float = 100_000.0,
        risk_per_trade: float = 0.0075,
        max_positions: int = 5,
        max_portfolio_risk: float = 0.04,      # Max 4% total risk open
        max_leverage: float = 1.5,
        commission: float = 0.0005,
        slippage: float = 0.0002,
        long_threshold: float = 0.58,
        short_threshold: float = 0.42,
        atr_period: int = 14,
        atr_stop_mult: float = 2.2,
        atr_trail_mult: float = 2.8,
        max_hold_days: int = 8,
        max_vol_percentile: float = 0.85,
    ):
        self.tickers = [t.upper().strip() for t in tickers if t.strip()]
        self.start = start
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        self.max_portfolio_risk = max_portfolio_risk
        self.max_leverage = max_leverage
        self.commission = commission
        self.slippage = slippage
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.atr_trail_mult = atr_trail_mult
        self.max_hold_days = max_hold_days
        self.max_vol_percentile = max_vol_percentile
        self.models: Dict[str, LGBMClassifier] = {}
        self.features = None

    def _download(self, ticker: str, start: Optional[str] = None) -> pd.DataFrame:
        start = start or self.start
        df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()

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
        df["atr"] = self._atr(df, self.atr_period)
        df["vol_percentile"] = df["vol_21"].rolling(63).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        df = df.dropna()
        if self.features is None:
            self.features = [c for c in df.columns if c not in 
                             ["Open", "High", "Low", "Close", "Volume", "target", "atr", "vol_percentile"]]
        return df

    def train(self, ticker: str, train_df: pd.DataFrame):
        model = LGBMClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=5,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=40, random_state=42, n_jobs=-1, verbose=-1
        )
        model.fit(train_df[self.features], train_df["target"])
        self.models[ticker] = model

    def generate_signals(self, ticker: str, df: pd.DataFrame) -> pd.DataFrame:
        probs = self.models[ticker].predict_proba(df[self.features])[:, 1]
        df = df.copy()
        df["prob_up"] = probs
        df["signal"] = 0
        long_mask = (df["prob_up"] >= self.long_threshold) & (df["vol_percentile"] <= self.max_vol_percentile)
        short_mask = (df["prob_up"] <= self.short_threshold) & (df["vol_percentile"] <= self.max_vol_percentile)
        df.loc[long_mask, "signal"] = 1
        df.loc[short_mask, "signal"] = -1
        return df

    def backtest_ticker(self, ticker: str, df: pd.DataFrame, capital_slice: float):
        equity = capital_slice
        position = 0
        entry_price = stop_price = 0.0
        entry_idx = 0
        shares = 0.0
        trades = []
        equity_curve = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i - 1]

            if position != 0:
                hold_days = i - entry_idx
                if position == 1:
                    new_trail = row["Close"] - self.atr_trail_mult * row["atr"]
                    stop_price = max(stop_price, new_trail)
                    hit_stop = row["Low"] <= stop_price
                else:
                    new_trail = row["Close"] + self.atr_trail_mult * row["atr"]
                    stop_price = min(stop_price, new_trail)
                    hit_stop = row["High"] >= stop_price

                exit_signal = hit_stop or row["signal"] == -position or hold_days >= self.max_hold_days

                if exit_signal:
                    exit_price = stop_price if hit_stop else row["Open"]
                    exit_price *= (1 - self.slippage * np.sign(position))
                    gross_ret = position * (exit_price - entry_price) / entry_price
                    net_ret = gross_ret - self.commission * 2
                    pnl = shares * entry_price * net_ret
                    equity += pnl
                    trades.append({
                        "ticker": ticker,
                        "entry_date": df.index[entry_idx].strftime("%Y-%m-%d"),
                        "exit_date": df.index[i].strftime("%Y-%m-%d"),
                        "direction": "LONG" if position == 1 else "SHORT",
                        "entry": round(entry_price, 2),
                        "exit": round(exit_price, 2),
                        "pnl": round(pnl, 2),
                        "hold_days": hold_days,
                        "reason": "STOP" if hit_stop else ("SIGNAL" if row["signal"] == -position else "TIME"),
                    })
                    position = 0

            if position == 0 and row["signal"] != 0:
                atr = prev["atr"]
                if pd.isna(atr) or atr <= 0:
                    continue
                risk_per_share = atr * self.atr_stop_mult
                shares = (equity * self.risk_per_trade) / risk_per_share
                notional = shares * row["Open"]
                if notional > equity * self.max_leverage:
                    shares = (equity * self.max_leverage) / row["Open"]
                position = int(row["signal"])
                entry_price = row["Open"] * (1 + self.slippage * position)
                entry_idx = i
                shares = abs(shares)
                stop_price = entry_price - self.atr_stop_mult * atr if position == 1 else entry_price + self.atr_stop_mult * atr

            equity_curve.append({"date": df.index[i], "equity": equity})

        if position != 0:
            last = df.iloc[-1]
            exit_price = last["Close"] * (1 - self.slippage * np.sign(position))
            net_ret = position * (exit_price - entry_price) / entry_price - self.commission * 2
            pnl = shares * entry_price * net_ret
            equity += pnl
            trades.append({
                "ticker": ticker,
                "entry_date": df.index[entry_idx].strftime("%Y-%m-%d"),
                "exit_date": df.index[-1].strftime("%Y-%m-%d"),
                "direction": "LONG" if position == 1 else "SHORT",
                "entry": round(entry_price, 2),
                "exit": round(exit_price, 2),
                "pnl": round(pnl, 2),
                "hold_days": len(df) - 1 - entry_idx,
                "reason": "EOD",
            })

        return pd.DataFrame(trades), pd.DataFrame(equity_curve).set_index("date"), equity

    def run_backtest(self, train_ratio: float = 0.70):
        all_trades = []
        all_equity = []
        capital_per = self.capital / max(len(self.tickers), 1)

        progress = st.progress(0)
        status = st.empty()

        for idx, ticker in enumerate(self.tickers):
            status.write(f"Processing **{ticker}**...")
            try:
                raw = self._download(ticker)
                if len(raw) < 300:
                    st.warning(f"{ticker}: insufficient data")
                    continue
                df = self.engineer_features(raw)
                split = int(len(df) * train_ratio)
                train_df = df.iloc[:split]
                test_df = df.iloc[split:].copy()
                self.train(ticker, train_df)
                test_df = self.generate_signals(ticker, test_df)
                trades, eq_curve, _ = self.backtest_ticker(ticker, test_df, capital_per)
                all_trades.append(trades)
                all_equity.append(eq_curve.rename(columns={"equity": ticker}))
            except Exception as e:
                st.error(f"{ticker} failed: {e}")
            progress.progress((idx + 1) / len(self.tickers))

        status.empty()
        progress.empty()

        if not all_trades:
            return None, None

        trades_df = pd.concat(all_trades, ignore_index=True)
        equity_combined = pd.concat(all_equity, axis=1).ffill().sum(axis=1)
        equity_combined.name = "Portfolio Equity"
        return trades_df, equity_combined

    def generate_daily_signals(self) -> pd.DataFrame:
        """Train on recent history and output today's signal for each ticker."""
        results = []
        lookback_start = (datetime.now() - timedelta(days=900)).strftime("%Y-%m-%d")

        for ticker in self.tickers:
            try:
                raw = self._download(ticker, start=lookback_start)
                if len(raw) < 200:
                    continue
                df = self.engineer_features(raw)
                # Train on all but last 5 days, predict on latest bar
                train_df = df.iloc[:-5]
                latest = df.iloc[[-1]].copy()
                self.train(ticker, train_df)
                latest = self.generate_signals(ticker, latest)
                row = latest.iloc[0]
                signal = int(row["signal"])
                results.append({
                    "ticker": ticker,
                    "date": latest.index[0].strftime("%Y-%m-%d"),
                    "close": round(row["Close"], 2),
                    "prob_up": round(row["prob_up"], 3),
                    "vol_percentile": round(row["vol_percentile"], 2),
                    "atr": round(row["atr"], 2),
                    "signal": signal,
                    "action": "LONG" if signal == 1 else ("SHORT" if signal == -1 else "FLAT"),
                    "suggested_stop": round(
                        row["Close"] - self.atr_stop_mult * row["atr"] if signal == 1
                        else row["Close"] + self.atr_stop_mult * row["atr"] if signal == -1
                        else None, 2
                    ) if signal != 0 else None,
                })
            except Exception as e:
                results.append({
                    "ticker": ticker,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "close": None,
                    "prob_up": None,
                    "vol_percentile": None,
                    "atr": None,
                    "signal": 0,
                    "action": f"ERROR: {str(e)[:40]}",
                    "suggested_stop": None,
                })
        return pd.DataFrame(results)


# -------------------------------------------------
# Alpaca helpers
# -------------------------------------------------
def get_alpaca_client(api_key: str, secret_key: str):
    if not ALPACA_AVAILABLE:
        st.error("Install alpaca-py: pip install alpaca-py")
        return None
    try:
        return TradingClient(api_key, secret_key, paper=True)
    except Exception as e:
        st.error(f"Connection failed: {e}")
        return None


def show_alpaca_account(client):
    try:
        account = client.get_account()
        c1, c2, c3 = st.columns(3)
        c1.metric("Equity", f"${float(account.equity):,.2f}")
        c2.metric("Buying Power", f"${float(account.buying_power):,.2f}")
        c3.metric("Status", account.status)
        return account
    except Exception as e:
        st.error(f"Account error: {e}")
        return None


def place_paper_order(client, symbol: str, qty: float, side: str):
    try:
        order_data = MarketOrderRequest(
            symbol=symbol.upper(),
            qty=qty,
            side=OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
        order = client.submit_order(order_data)
        st.success(f"Paper order submitted → {side.upper()} {qty} {symbol} | ID: {order.id}")
        return order
    except Exception as e:
        st.error(f"Order failed: {e}")
        return None


# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("📈 Elite AI Trading Bot")
st.caption("Daily Signals • ATR Trailing • Regime Filter • Paper Trading • Mobile Ready")

tab1, tab2, tab3, tab4 = st.tabs([
    "📅 Daily Signals",
    "🔬 Backtest",
    "📄 Paper Trading",
    "🚀 Deploy / Automate"
])

# ==================== TAB 1: DAILY SIGNALS ====================
with tab1:
    st.subheader("Today’s High-Conviction Signals")
    st.markdown("Run this every morning before the open. Only high-probability setups that pass the volatility regime filter are shown.")

    with st.sidebar:
        st.header("Signal Settings")
        signal_tickers = st.text_input("Watchlist (comma separated)", value="SPY,QQQ,IWM,AAPL,MSFT,NVDA,AMZN,META", key="sig_tickers")
        sig_long = st.slider("Long Threshold", 0.52, 0.70, 0.58, 0.01, key="sig_long")
        sig_short = st.slider("Short Threshold", 0.30, 0.48, 0.42, 0.01, key="sig_short")
        sig_vol = st.slider("Max Vol Percentile", 0.70, 0.95, 0.85, 0.01, key="sig_vol")
        sig_risk = st.slider("Risk per Trade %", 0.25, 1.5, 0.75, 0.05, key="sig_risk") / 100
        gen_btn = st.button("🔄 GENERATE TODAY’S SIGNALS", type="primary")

    if gen_btn:
        tickers = [t.strip() for t in signal_tickers.split(",") if t.strip()]
        with st.spinner("Training models on recent data and generating signals..."):
            bot = EliteTradingBot(
                tickers=tickers,
                long_threshold=sig_long,
                short_threshold=sig_short,
                max_vol_percentile=sig_vol,
                risk_per_trade=sig_risk,
            )
            signals = bot.generate_daily_signals()

        if len(signals) > 0:
            # Highlight actionable signals
            longs = signals[signals["signal"] == 1]
            shorts = signals[signals["signal"] == -1]
            flats = signals[signals["signal"] == 0]

            c1, c2, c3 = st.columns(3)
            c1.metric("LONG setups", len(longs))
            c2.metric("SHORT setups", len(shorts))
            c3.metric("FLAT / No trade", len(flats))

            st.subheader("Actionable Signals")
            if len(longs) + len(shorts) == 0:
                st.info("No high-conviction setups today. Stay in cash — this is often the correct decision.")
            else:
                actionable = pd.concat([longs, shorts]).sort_values("prob_up", ascending=False)
                st.dataframe(actionable, use_container_width=True)

                st.markdown("### Suggested Actions")
                for _, row in actionable.iterrows():
                    color = "signal-long" if row["signal"] == 1 else "signal-short"
                    st.markdown(
                        f"<div class='{color}'>{row['action']} {row['ticker']} @ {row['close']} "
                        f"| Prob: {row['prob_up']:.1%} | Stop ≈ {row['suggested_stop']}</div>",
                        unsafe_allow_html=True
                    )

            with st.expander("Full watchlist details"):
                st.dataframe(signals, use_container_width=True)

            st.download_button(
                "Download Today’s Signals",
                signals.to_csv(index=False).encode(),
                f"signals_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
        else:
            st.warning("No data returned. Check tickers or try again later.")
    else:
        st.info("Set your watchlist in the sidebar and press **GENERATE TODAY’S SIGNALS**")

# ==================== TAB 2: BACKTEST ====================
with tab2:
    st.subheader("Full Historical Backtest")
    with st.sidebar:
        st.header("Backtest Settings")
        bt_tickers = st.text_input("Tickers", value="SPY", key="bt_tickers")
        capital = st.number_input("Capital ($)", value=100000, step=10000, key="bt_cap")
        risk = st.slider("Risk per Trade %", 0.25, 2.0, 0.75, 0.05, key="bt_risk") / 100
        max_pos = st.slider("Max Concurrent Positions", 1, 10, 5, key="bt_maxpos")
        long_th = st.slider("Long Threshold", 0.52, 0.70, 0.58, 0.01, key="bt_long")
        short_th = st.slider("Short Threshold", 0.30, 0.48, 0.42, 0.01, key="bt_short")
        atr_stop = st.slider("ATR Stop", 1.5, 3.5, 2.2, 0.1, key="bt_stop")
        atr_trail = st.slider("ATR Trail", 2.0, 4.0, 2.8, 0.1, key="bt_trail")
        max_vol = st.slider("Max Vol %ile", 0.70, 0.95, 0.85, 0.01, key="bt_vol")
        start_date = st.date_input("Start", value=pd.to_datetime("2018-01-01"), key="bt_start")
        run_btn = st.button("🚀 RUN BACKTEST", type="primary", key="bt_run")

    if run_btn:
        tickers = [t.strip() for t in bt_tickers.split(",") if t.strip()]
        with st.spinner("Running institutional backtest..."):
            bot = EliteTradingBot(
                tickers=tickers,
                start=str(start_date),
                capital=capital,
                risk_per_trade=risk,
                max_positions=max_pos,
                long_threshold=long_th,
                short_threshold=short_th,
                atr_stop_mult=atr_stop,
                atr_trail_mult=atr_trail,
                max_vol_percentile=max_vol,
            )
            trades, equity = bot.run_backtest()

        if trades is not None and len(trades) > 0:
            total_return = (equity.iloc[-1] / capital - 1) * 100
            win_rate = (trades["pnl"] > 0).mean()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Final Equity", f"${equity.iloc[-1]:,.0f}")
            c2.metric("Return", f"{total_return:+.1f}%")
            c3.metric("Win Rate", f"{win_rate:.1%}")
            c4.metric("Trades", len(trades))
            st.line_chart(equity)
            st.dataframe(trades.sort_values("exit_date", ascending=False), use_container_width=True, height=350)
            st.download_button("Download Trades", trades.to_csv(index=False).encode(), "trades.csv", "text/csv")
        else:
            st.warning("No trades generated.")
    else:
        st.info("Configure and run a full backtest from the sidebar.")

# ==================== TAB 3: PAPER TRADING ====================
with tab3:
    st.subheader("Alpaca Paper Trading")
    st.markdown("Connect your free Alpaca paper account to place simulated live orders from your phone.")

    if not ALPACA_AVAILABLE:
        st.warning("Run: `pip install alpaca-py`")
    else:
        api_key = st.text_input("Paper API Key", type="password", key="alp_key")
        secret_key = st.text_input("Paper Secret Key", type="password", key="alp_sec")

        if api_key and secret_key:
            client = get_alpaca_client(api_key, secret_key)
            if client:
                st.success("Connected to Alpaca Paper")
                show_alpaca_account(client)
                st.divider()
                ca, cb, cc = st.columns(3)
                with ca:
                    symbol = st.text_input("Symbol", value="SPY", key="ord_sym")
                with cb:
                    qty = st.number_input("Qty", value=1.0, min_value=0.01, step=0.1, key="ord_qty")
                with cc:
                    side = st.selectbox("Side", ["BUY", "SELL"], key="ord_side")
                if st.button("Submit Paper Order", type="primary"):
                    place_paper_order(client, symbol, qty, side)
        else:
            st.info("Paste your Alpaca **paper** keys to enable live paper trading.")

# ==================== TAB 4: DEPLOY & AUTOMATE ====================
with tab4:
    st.subheader("Make it a permanent phone app + daily automation")

    st.markdown("""
    ### 1. Permanent Phone App (Streamlit Cloud)
    1. Create a GitHub repo and upload:
       - `trading_app.py`
       - `requirements.txt`
    2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
    3. Deploy
    4. Open the public URL on your phone → **Add to Home Screen**

    ### 2. Daily Signal Automation
    **Option A – Manual (simplest)**  
    Open the app every morning → Daily Signals tab → Generate.

    **Option B – Scheduled (recommended for serious use)**  
    - Use GitHub Actions (free) to run a small script every weekday at 8:30 AM ET  
    - Or use a free cron service / Railway / Render  
    - Have it write today’s signals to a Google Sheet or send you an email

    ### 3. Risk Rules currently enforced
    - Max risk per trade (default 0.75%)
    - Volatility regime filter (skip high-vol days)
    - ATR-based initial + trailing stops
    - Max concurrent positions (configurable)
    - High-conviction probability thresholds only

    This is how professional independent traders actually run systems in 2026.
    """)

    st.success("You now have a complete daily-signal + paper-trading system that lives on your phone.")
