from binance import AsyncClient, BinanceSocketManager
import asyncio
import time

import pandas as pd

from binance import ThreadedWebsocketManager
from binance.client import Client
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

    await client.close_connection()


loop = asyncio.get_event_loop()
loop.run_until_complete(a())
