# main.py
from src.data import download_stock_data, preprocess_data
from src.model import train_model, create_features
from src.evaluate import evaluate_predictions, filter_top_performers
import pandas as pd

# Configuratie
SYMBOLS = ["AAPL", "MSFT", "GOOGL"]  # Uitbreidbaar met meer aandelen
YEARS_HISTORY = 21

results = []

for symbol in SYMBOLS:
    # Stap 1: Data ophalen en verwerken
    raw_data = download_stock_data(symbol, YEARS_HISTORY)
    processed_data = preprocess_data(raw_data)
    
    # Stap 2: Model trainen
    X, y = create_features(processed_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = train_model(X_train, y_train)
    
    # Stap 3: Voorspellingen evalueren
    predictions = model.predict(X_test)
    accuracy = evaluate_predictions(y_test, predictions)
    
    results.append({"symbol": symbol, "accuracy": accuracy})

# Filter beste resultaten
top_performers = filter_top_performers(pd.DataFrame(results))
print("Beste aandelen:", top_performers)