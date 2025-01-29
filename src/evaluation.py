from sklearn.metrics import classification_report

print(classification_report(test['Target'], model.predict(test[['SMA_50', 'RSI']])))