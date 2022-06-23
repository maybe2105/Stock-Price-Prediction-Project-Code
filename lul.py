from binance import AsyncClient, BinanceSocketManager
import asyncio
import numpy as np
import time

import pandas as pd

from binance import ThreadedWebsocketManager
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

scaler = MinMaxScaler(feature_range=(0, 1))

api_key = 'NwwA7xRmAq0juYxurtfqmAiW7asFDxB33zQknnaOEmEItWnbR0bVVtjoZLV6tQBy'
api_secret = 'Lqq5JzhBANnszyHblQwRhKDgrPjfmjDGdcZA7BZdg3e7pJZkir5vRnBTQh8k4ypp'
apiClient = Client(api_key, api_secret)


df = pd.DataFrame(apiClient.get_historical_klines(
    "BTCUSDT", Client.KLINE_INTERVAL_1MINUTE))
df = df.iloc[:, :6]
df.columns = ["Date", "o", "h", "l", "c", "v"]
df = df.set_index("Date")
df.index = pd.to_datetime(df.index, unit="ms")
df = df.astype("float")

data = df.to_dict("records")
# print(call)

# data = []
model = load_model("saved_rnn.h5")


def get_predicted_val(data):
    val = data[-60:].values
    scaled_data = scaler.fit_transform(val)
    X_test = []
    X_test.append(scaled_data)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)
    return pred_price


def get_predicted_price(data):
    df = pd.DataFrame.from_records(data)
    close = get_predicted_val(df.filter(['c']))
    open = get_predicted_val(df.filter(['o']))
    high = get_predicted_val(df.filter(['h']))
    low = get_predicted_val(df.filter(['l']))
    print(close[0], open[0], high[0], low[0])
    # return close, open, high, low


async def a():
    client = await AsyncClient.create()
    bm = BinanceSocketManager(client)
    # start any sockets here, i.e a trade socket
    ts = bm.symbol_miniticker_socket("BTCUSDT")
    # then start receiving messages
    async with ts as tscm:
        while True:
            res = await tscm.recv()
            data.append(res)
            get_predicted_price(data)

    await client.close_connection()


loop = asyncio.get_event_loop()
loop.run_until_complete(a())
