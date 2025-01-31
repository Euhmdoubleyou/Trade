# src/evaluate.py
import pandas as pd

def evaluate_predictions(true_values, predictions):
    """Evalueer de nauwkeurigheid van voorspellingen."""
    accuracy = (true_values == predictions).mean()
    return accuracy

def filter_top_performers(results, threshold=0.6):
    """Filter aandelen met een nauwkeurigheid boven een drempel."""
    return results[results["accuracy"] > threshold]