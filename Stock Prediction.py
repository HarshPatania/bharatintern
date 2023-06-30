import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

data = pd.read_csv('C:\\Users\\Harsh\\OneDrive\\Desktop\\Bharat Intern\\archive\\ADANIPORTS.csv')

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length - 1):
        X.append(data[i:(i + sequence_length), 0])
        y.append(data[i + sequence_length, 0])
    return np.array(X), np.array(y)

sequence_length = 20 
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=10, batch_size=16)

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_predictions = scaler.inverse_transform(train_predictions)
y_train = scaler.inverse_transform([y_train])
test_predictions = scaler.inverse_transform(test_predictions)
y_test = scaler.inverse_transform([y_test])

plt.plot(np.arange(len(train_predictions)), y_train.flatten(), label='Actual Train')
plt.plot(np.arange(len(train_predictions)), train_predictions.flatten(), label='Predicted Train')
plt.plot(np.arange(len(train_predictions), len(train_predictions) + len(test_predictions)), y_test.flatten(), label='Actual Test')
plt.plot(np.arange(len(train_predictions), len(train_predictions) + len(test_predictions)), test_predictions.flatten(), label='Predicted Test')
plt.legend()
plt.show()
