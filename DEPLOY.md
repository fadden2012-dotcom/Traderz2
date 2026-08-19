# Deploy Elite AI Trading Bot as a Phone App

## Files you need
- trading_app.py
- requirements.txt

## 1. Local run (test first)
```bash
pip install -r requirements.txt
streamlit run trading_app.py
```

## 2. Deploy to Streamlit Cloud (permanent phone access)

1. Create a free GitHub repository
2. Upload `trading_app.py` and `requirements.txt`
3. Go to https://share.streamlit.io and sign in with GitHub
4. Click **New app**
5. Select your repository → Main file path: `trading_app.py`
6. Click **Deploy**

After 1–2 minutes you get a public URL.

Open that URL on your phone → browser menu → **Add to Home Screen**.

You now have a permanent trading app icon on your phone.

## 3. Daily Signals
Use the **Daily Signals** tab every morning.
It trains on recent data and only shows high-conviction setups that pass the volatility regime filter.

## 4. Alpaca Paper Trading
1. Create free account at https://alpaca.markets
2. Generate Paper Trading API Key + Secret
3. Paste them in the Paper Trading tab
4. Place simulated live orders from your phone

Never use live keys until you are fully ready.
