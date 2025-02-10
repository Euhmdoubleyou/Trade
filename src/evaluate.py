import pandas as pd
from sklearn.metrics import accuracy_score
import os

def evaluate_predictions(true_directions, predicted_directions):
    """Evalueer het percentage correcte richtingsvoorspellingen (↑/↓)."""
    accuracy = accuracy_score(true_directions, predicted_directions)
    return accuracy

""" def filter_top_performers(results, threshold=0.7):
    Filter aandelen met een nauwkeurigheid boven de drempel.
    return results[results["accuracy"] > threshold]"""

def check_symbol(symbol, true_directions, predicted_directions, threshold):
    """Checkt of een symbool aan de drempel voldoet."""
    accuracy = evaluate_predictions(true_directions, predicted_directions)
    print(f"Symbool: {symbol}, Nauwkeurigheid: {accuracy:.2f}")
    
    if accuracy > threshold:
        print(f"{symbol} voldoet aan de drempel van {threshold*100}%")
        return (symbol, true_directions, predicted_directions, threshold)

        
    else:
        print(f"{symbol} voldoet niet aan de drempel van {threshold*100}%")
        
    remove_raw_data(symbol)

def remove_raw_data(symbol):
    #"""Verwijdert het .csv bestand uit de data\raw map."""
    file_path = f"data/raw/{symbol}.csv"
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{symbol}.csv verwijderd uit data/raw")
    else:
        print(f"{symbol}.csv bestaat niet in data/raw")

def main():
    # Implementeer de logica om symbolen te checken en te filteren
    pass


def calculate_consecutive_up_days(prices):
    """Bereken het maximale aantal opeenvolgende stijgende dagen."""
    up_days = (prices > prices.shift(1)).astype(int)
    streaks = up_days * (up_days.groupby((up_days != up_days.shift()).cumsum()).cumcount() + 1)
    return streaks.max()

if __name__ == "__main__":
    main()
