import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import pandas as pd
import pandas_datareader as web
import datetime as dt
import math

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
plt.style.use('fivethirtyeight')

df = web.DataReader('GOOG', data_source='yahoo',
                    start='2021-06-07', end='2022-06-07')

# need to make into numpy arrays because only nump arrays can be input values in keras
data = df.filter(['Close'])
dataset = data.values
training_data_len = math.ceil(len(dataset) * .8)

# scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:training_data_len, :]


prediction_days = 60

x_train = []
y_train = []
for x in range(prediction_days, len(train_data)):
    x_train.append(train_data[x-prediction_days:x, 0])
    y_train.append(train_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

regressor = Sequential()

# RNN
# regressor- object of sequential class, can add layers to networ.
regressor.add(LSTM(units=96, return_sequences=True,
                   input_shape=(x_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=96, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=96, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=96))
regressor.add(Dropout(0.2))
regressor.add(Dense(units=1))
regressor.compile(optimizer='adam', loss='mean_squared_error')
# CHANGE BACK TO 100 IF NOT WORK
regressor.fit(x_train, y_train, epochs=50, batch_size=32)


# LTSM
# regressor.add(LSTM(50, return_sequences=True,
#               input_shape=(x_train.shape[1], 1)))
# regressor.add(LSTM(50, return_sequences=False))
# regressor.add(Dense(25))
# regressor.add(Dense(1))
# regressor.compile(optimizer='adam', loss='mean_squared_error')
# regressor.fit(x_train, y_train, batch_size=1, epochs=1)

test_data = scaled_data[training_data_len - 60:, :]


X_test = []
Y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

rmse = np.sqrt(np.mean((predicted_stock_price - Y_test)**2))


regressor.save('saved_rnn.h5')

train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predicted_stock_price


apple_quote = web.DataReader(
    'GOOG', data_source='yahoo', start='2021-07-06', end='2022-07-06')
# create a new dataframe
new_df = apple_quote.filter(['Close'])
# get the last 60 day closing price values and convert the dataframe to an array
last_60_days = new_df[-60:].values
# scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
# create an empty list
X_test = []
# append the past 60 days
X_test.append(last_60_days_scaled)
# convert the X_test data to a numpy array
X_test = np.array(X_test)
# reshape
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# get the predicted scaled price
pred_price = regressor.predict(X_test)
# undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)

plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
