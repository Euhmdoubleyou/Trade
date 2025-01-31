# src/model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model(X, y):
    """Train een model om stijgingen te voorspellen."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def create_features(data):
    """Maak features op basis van historische data."""
    X = data[["Daily_Return"]].shift(1).dropna()  # Gebruik rendement van gisteren
    y = (data["Daily_Return"] > 0).astype(int).shift(-1).dropna()  # Target: stijging morgen
    X = X.iloc[:-1]  # Align X en y
    return X, y