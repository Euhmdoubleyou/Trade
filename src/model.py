# src/model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model(X, y):
    """Train een model om stijgingen te voorspellen."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def create_features_optimized(data):
    """Maak geavanceerde features op basis van historische data."""
    # Technische indicatoren
    data['MA_5'] = data['Close'].rolling(5).mean().shift(1)
    data['MA_20'] = data['Close'].rolling(20).mean().shift(1)
    data['MA_50'] = data['Close'].rolling(50).mean().shift(1)
    data['Volatility'] = data['Close'].rolling(20).std().shift(1)
    data['Momentum'] = data['Close'].pct_change(5).shift(1)
    
    # Lagged returns
    for lag in [1, 2, 3, 5]:
        data[f'Return_lag_{lag}'] = data['Close'].pct_change(lag).shift(1)
    
    # Target: Stijging morgen (1) of daling (0)
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    X = data.dropna().drop(['Target', 'Close'], axis=1)
    y = data.dropna()['Target']
    return X, y