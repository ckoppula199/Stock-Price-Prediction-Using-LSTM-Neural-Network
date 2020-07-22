"""
Using an artificial recurrent neural network (LSTM) this program attempts to
predict the closing stock price of a corporation.
"""

import pandas_datareader as pd_data
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

STOCK_SYMBOL = 'AAPL'

# Get data for desired company within specified time frame
# Can be changed to anlyse different data
df = pd_data.DataReader(f"{STOCK_SYMBOL}", data_source='yahoo', start='2013-01-01', end='2020-07-1')
print("\n\nView structure of data")
print(df.head(), end='\n\n\n')

# Find out the number of rows and columns in the df
print("(Rows, Columns)")
print(df.shape, end='\n\n\n')

# Visualise the closing price history
plt.figure(figsize=(16,8))
plt.title("Close Price History")
# Index of df in the date so date is used as the x-axis
plt.plot(df.index, df['Close'])
plt.xlabel("Date")
plt.ylabel("Close Price USD ($)")
plt.show()

# Create new dataframe only containing close column and convert to np array
data = df.filter(['Close'])
dataset = data.values
# find number of rows of data to train model on (80% training data)
training_data_len = math.ceil(len(dataset) * 0.8)
print(f"Amount of training data is {training_data_len}", end='\n\n\n')

# Split into training data
train_data = dataset[:training_data_len]

# Apply feature scalling to the data for improved efficiency
scaler = MinMaxScaler(feature_range=(0, 1))
train_data = scaler.fit_transform(train_data)

# split training data into x_train and y_train data sets
x_train = []
y_train = []

# Use last 60 days close price
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data to be 3-dimensional which is what the LSTM expects
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Building the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Generate testing data
test_data = scaler.transform(dataset[training_data_len - 60:, :])

x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)

# Reshape the data to be 3-dimensional which is what the LSTM expects
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get models predicted price values
y_pred = model.predict(x_test)
y_pred = scaler.inverse_transform(y_pred)

# Get RMSE
rmse = mean_squared_error(y_test, y_pred)
print(f"\n\n\nroot mean squared error is {rmse}", end='\n\n\n')

# Plot the data
train = data[:training_data_len]
actual = data[training_data_len:]
actual['Predictions'] = y_pred
plt.figure(figsize=(16,8))
plt.title("Model predictions")
plt.xlabel("Date")
plt.ylabel("Close Price USD ($)")
plt.plot(train['Close'])
plt.plot(actual[['Close', 'Predictions']])
plt.legend(['Train', 'Actual',  'Predictions'], loc='lower right')
plt.show()

# Predict stock price one day later
# Can form a markov chain to predict mutliple days in the future
data_to_predict = pd_data.DataReader(f"{STOCK_SYMBOL}", data_source='yahoo', start='2020-01-01', end='2020-07-1')
predict_df = data_to_predict.filter(['Close'])
# Get last 60 days close price into array format
last_60_days = predict_df[-60:].values
# Scale the data
last_60_days = scaler.transform(last_60_days)
x_predict_test = np.array([last_60_days])
# Reshape the data
x_predict_test = np.reshape(x_predict_test, (x_predict_test.shape[0], x_predict_test.shape[1], 1))
# Make predictions and undo the scaling
pred_price = model.predict(x_predict_test)
pred_price = scaler.inverse_transform(pred_price)

# Get actual price
actual_price = pd_data.DataReader(f"{STOCK_SYMBOL}", data_source='yahoo', start='2020-07-2', end='2020-07-2')['Close'].values
print(f"Predicted price: {pred_price[0][0]}", end='\n\n')
print(f"Actual price: {actual_price[0]}")
