# main.py
from src.data import download_stock_data, preprocess_data
from src.model import train_model, create_features
from src.evaluate import evaluate_predictions, filter_top_performers, check_symbol, remove_raw_data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

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
    if len(X) != len(y):
    # Handle the length mismatch here, e.g., by dropping extra rows
        X = X[:len(y)]
        y = y[:len(X)]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = train_model(X_train, y_train)
    
    param_grid = { ... }  # Zie voorbeeld hierboven
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    # Stap 3: Voorspellingen evalueren
    predictions = model.predict(X_test)
    true_directions = (y_test > y_test.shift(1)).fillna(0).astype(int)
    predicted_directions = (predictions > predictions.mean()).astype(int)
    accuracy = evaluate_predictions(y_test, predictions)
    
     # Oproepen van check_symbol methode
    check_symbol(symbol, true_directions, predicted_directions, threshold=0.8)

    results.append({"symbol": symbol, "accuracy": accuracy})

# Filter beste resultaten
top_performers = filter_top_performers(pd.DataFrame(results))
print("Beste aandelen:", top_performers)


