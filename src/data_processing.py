def clean_data(df):
    # Handle missing values
    df = df.ffill()  # Forward-fill missing data
    df = df.dropna()
    
    # Verwijder duplicates
    df = df[~df.index.duplicated()]
    
    # Sorteer op datum
    df = df.sort_index()
    
    return df

data = clean_data(data)

# Voorbeeld: Technische indicatoren
data['SMA_50'] = data['Close'].rolling(window=50).mean()  # Moving Average
data['RSI'] = ...  # Relative Strength Index (gebruik talib als library)
data['Daily_Return'] = data['Close'].pct_change()  # Dagelijkse rendementen