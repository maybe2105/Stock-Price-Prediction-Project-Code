import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import math
import pandas_datareader as web

df = web.DataReader('GOOG', data_source='yahoo',
                    start='2021-06-07', end='2022-06-07')

data = df.filter(['Close'])
dataset = data.values
training_data_len = math.ceil(len(dataset) * .8)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
dataset_test = scaler.transform(dataset)

dataset_train = scaled_data[0:training_data_len, :]


def create_dataset(df):
    x = []
    y = []
    for i in range(60, df.shape[0]):
        x.append(df[i-60:i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x, y


x_train, y_train = create_dataset(dataset_train)
x_test, y_test = create_dataset(dataset_test)

model = Sequential()
model.add(LSTM(units=96, return_sequences=True,
          input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=96, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96))
model.add(Dropout(0.2))
model.add(Dense(units=1))


x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(x_train, y_train, epochs=40, batch_size=32)
model.save('rnn_model.h5')


model = load_model('rnn_model.h5')
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

print("==========================")
print(predictions)

fig, ax = plt.subplots(figsize=(16, 8))
ax.set_facecolor('#000041')
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD($)', fontsize=18)
plt.plot(predictions, color='cyan', label='Predicted price')
ax.plot(y_test_scaled, color='red', label='Original price')


plt.show()
