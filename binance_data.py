import time
from datetime import datetime

from kline import Kline


# use binance client!

def get_klines(client, interval, num_klines, symbol):

    klines = []

    remaining_klines = num_klines

    abort = False

    while remaining_klines > 0 and not abort:

        limit = 1000 if remaining_klines >= 1000 else (remaining_klines) % 1000

        end_time = int(time.time() * 1000) if len(
            klines) == 0 else klines[0].openTime

        next = [Kline(kline) for kline in client.get_klines(
            symbol=symbol, interval=interval, endTime=end_time - (1000 * 60 * 60 * 12), limit=limit)]

        if len(next) == 0:
            print(
                f"no klines before {datetime.fromtimestamp(klines[0].openTime / 1000)}")
            abort = True

        klines = next + klines

        remaining_klines -= 1000
        time.sleep(0.2)

    return klines


def assert_klines(klines, interval):
    for n in range(1, len(klines)):
        if klines[n].closeTime - klines[n-1].closeTime != interval:
            print(
                f"Kline intarvals not right at from index {n-1} to index {n}")
            print(
                f"goes from {datetime.fromtimestamp(klines[n-1].closeTime / 1000)} to {datetime.fromtimestamp(klines[n].closeTime / 1000)}")

def compute_returns_from_klines(klines):
    return [0] + [klines[i].avg / klines[i-1].avg - 1 for i in range(1, len(klines))]



# MAIN FUNCTION

'''
@return 
an array with dicts that have: 
    - 'close' as the close price
    - 'return' as the return to that time (starting with 1)
    - 'timestamp' self explanatory 
'''
def get_data(client, interval, limit, symbol):

    data = []

    klines = get_klines(client, interval, limit, symbol)
    returns = compute_returns_from_klines(klines)

    for i in range(len(klines)):

        d = {
            'close': klines[i].close,
            'return': returns[i],
            'timestamp': klines[i].closeTime,
        }

        data.append(d)

    return data