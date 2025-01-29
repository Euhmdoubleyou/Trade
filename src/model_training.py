from data_processing import data

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Target: "Gaat de prijs morgen omhoog?" (1 = ja, 0 = nee)
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

# Split data (let op: tijdreeks mag niet willekeurig gesplitst worden!)
train, test = train_test_split(data, test_size=0.2, shuffle=False)

# Train een simpel model
model = LogisticRegression()
model.fit(train[['SMA_50', 'RSI']], train['Target'])