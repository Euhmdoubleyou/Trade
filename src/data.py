# src/data.py
import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

def download_stock_data(symbol, years=21):
    """Download historische data voor een aandeel."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    end_date = pd.Timestamp.now() - pd.DateOffset(years=1)  # Tot 1 jaar geleden
    start_date = end_date - pd.DateOffset(years=years)
    
    data = yf.download(symbol, start=start_date, end=end_date)
    data.to_csv(DATA_DIR / f"{symbol}.csv")
    return data

def preprocess_data(data):
    """Basis data opschoning."""
    data = data.dropna()
    data = data[["Close"]]  # Alleen slotkoers gebruiken
    data["Daily_Return"] = data["Close"].pct_change()
    return data.dropna()