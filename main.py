# main.py
from src.data import download_stock_data, preprocess_data
from src.model import create_features_optimized, train_model
from src.evaluate import calculate_consecutive_up_days
from src.evaluate import evaluate_predictions, filter_top_performers, check_symbol, remove_raw_data
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from src.symbollist import symbollist
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

# Configuratie
# TryoutSymbols = ['QXO', 'PTITF', 'CSAN', 'CELH', 'MRNA',  'NVD'] # Om nieuwe code te proberen
SYMBOLS = symbollist # Zie symbollist.py voor selectie welke aandelen sector
YEARS_HISTORY = 21
results = []
Threshold = 0.7

for symbol in SYMBOLS:
    # Stap 1: Data ophalen en verwerken
    
    try: 
        raw_data = download_stock_data(symbol, YEARS_HISTORY)
        processed_data = preprocess_data(raw_data)
        
        # Stap 2: Model trainen
        X, y = create_features_optimized(processed_data)
        if len(X) != len(y):
        # Handle the length mismatch here, e.g., by dropping extra rows
            X = X[:len(y)]
            y = y[:len(X)]

        tscv = TimeSeriesSplit(n_splits=5)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = train_model(X_train, y_train)
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5],
            'max_features': ['sqrt', 'log2']
        }

        model = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        best_model = model.best_estimator_

        test_prices = processed_data.iloc[-len(y_test):]['Close']
        consecutive_up = calculate_consecutive_up_days(test_prices)
        # Stap 3: Voorspellingen evalueren
        predictions = model.predict(X_test)
        true_directions = (y_test > y_test.shift(1)).fillna(0).astype(int)
        predicted_directions = (predictions > predictions.mean()).astype(int)
        accuracy = evaluate_predictions(y_test, predictions)
        
        test_prices = processed_data.iloc[-len(y_test):]['Close']  # Prijzen in de testperiode
        max_consecutive_up = calculate_consecutive_up_days(test_prices)
        
        latest_prediction = model.predict(X.iloc[[-1]])  # Gebruik de laatste beschikbare data
        is_rising = "Ja" if latest_prediction[0] == 1 else "Nee"
        
        # Oproepen van check_symbol methode
        results.append({
        "symbol": symbol,
        "accuracy": accuracy,
        "stijging_verwacht": is_rising,
        "max_opvolgende_stijgingen_test": max_consecutive_up,
        })
        
        check_symbol(symbol, true_directions, predicted_directions, threshold=Threshold)
    except:
        print(f"niet genoeg data beschikbaar voor {symbol}")
        remove_raw_data(symbol)
        continue

# Filter beste resultaten
top_performers = filter_top_performers(pd.DataFrame(results), threshold=Threshold)
print("\nTop performers (nauwkeurigheid > 70%):")
print(top_performers[['symbol', 'accuracy', 'stijging_verwacht', 'max_opvolgende_stijgingen_test']])
""" for performer in results:
    print(calculate_consecutive_up_days(performer))
    
 """
