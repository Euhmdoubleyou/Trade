import yfinance as yf

symbol = 'AAPL'  # Apple Inc.
data = yf.download(symbol, start="2022-01-01", end="2023-12-01")
print(data)

data.columns = data.columns.droplevel(1)  # Verwijder 'AAPL' uit kolomnamen

data.to_csv("../data/raw/AAPL_historical.csv")