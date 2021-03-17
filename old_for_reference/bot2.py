from binance.client import Client

import binance_data as bd





def data_get(client, sources, targets, limit, interval):
    sources_data = []
    targets_data = []

    for symbol in sources:
        print(f"getting symbol {symbol}")
        data = bd.get_data(client, interval, limit, symbol)
        print(f"got {len(data)} entries for {symbol}")
        sources_data.append(data)

    for symbol in targets:
        print(f"getting symbol {symbol}")
        data = bd.get_data(client, interval, limit, symbol)
        print(f"got {len(data)} entries for {symbol}")
        targets_data.append(data)

def data_prepare(data, window_size):
    pass