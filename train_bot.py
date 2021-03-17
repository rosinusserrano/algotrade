#%%
# BINANCE
from binance.client import Client

# UTILS
import matplotlib.pyplot as plt
from pathlib import Path
from uuid import uuid4
import json

# TORCH
import torch
import torch.optim as optim

# OWN
import binance_data as bd
from tradenet import TradeNet

torch.set_default_dtype(torch.float64)

torch.set_printoptions(precision=10)


#%%

def buy_and_hold(validation_prices):
    
    num_markets = validation_prices.shape[1]
    length = validation_prices.shape[-1]
    money = torch.ones([num_markets,1]).double()
    money_l = [money]
    for i in range(length):
        money = money * (1.0 + validation_prices[0,:,i].reshape([num_markets, 1]))
        money_l.append(money)
    return money_l

#%%

def train_bot(client, source_symbols, target_symbols, limit, interval, window_size, comission, epochs, learning_rate, archive=False):
    
    if archive != False:
        # setup archive directory
        ## create random ID
        archive_dir = "archive/" + str(uuid4())
        ## create directory
        Path(archive_dir + "/plots").mkdir(parents=True, exist_ok=True)
        Path(archive_dir + "/weights").mkdir(parents=True, exist_ok=True)
        print(archive_dir)
        ## add info file
        info = {
            'source': source_symbols,
            'target': target_symbols,
            'limit': limit,
            'interval': str(interval),
            'window_size': window_size,
            'comission': comission,
        }
        fp = open(f"{archive_dir}/info.json", "w")
        json.dump(info, fp, sort_keys=True, indent=2)
        fp.close()

    #get data for source symbols
    source_currencies = []
    for symbol in source_symbols:
        print(f"getting symbol {symbol}")
        returns = torch.tensor([b['return'] for b in bd.get_data(client, interval, limit, symbol)])
        print(f"got {len(returns)} from {symbol}")
        source_currencies.append(returns)

    #get data for target symbols
    target_currencies = []
    for symbol in target_symbols:
        print(f"getting symbol {symbol}")
        returns = torch.tensor([b['return'] for b in bd.get_data(client, interval, limit, symbol)])
        print(f"got {len(returns)} from {symbol}")
        target_currencies.append(returns)

    #get minimum length of data to cut them in euqal lengths
    length = min([len(data) for data in source_currencies + target_currencies])
    split_length = max(int(length * 0.8), length - (1 * 4 * 24 * 7 * 4))    #save 80% of length for train val split (max 4 weeks)
    source_currencies = [returns[-length:] for returns in source_currencies]
    target_currencies = [returns[-length:] for returns in target_currencies]

    #make tensor from data
    source_prices = torch.stack(source_currencies).reshape([1, len(source_symbols), length]).double()
    target_prices = torch.stack(target_currencies).reshape([1, len(target_symbols), length]).double()

    net = TradeNet(source_symbols, target_symbols, window_size)

    net.double()

    source_train, source_validation = source_prices[:,:,:split_length], source_prices[:,:,split_length-window_size:]
    target_train, target_validation = target_prices[:,:,:split_length], target_prices[:,:,split_length-window_size:]

    print("\n\n\nTRAIN...")
    money_metric = net.train(source_train, target_train, source_validation, target_validation, comission, epochs, learning_rate)

    print("\n\n\nVALIDATE...")
    print(net.validate(source_validation, target_validation, comission))

    print("\n\n\nWALK...")
    money_walk, trade_points = net.walk(source_validation, target_validation, comission)

    print("\n\n\nBUY AND HOLD...")
    bah_money_walk = buy_and_hold(target_validation[:,:,window_size:])

    if False:
        print("RESULTS:")
        print("money metric")
        print(money_metric)
        print("\n")
        print("money walk")
        print(money_walk)
        print("\n")
        print("trade points")
        print(trade_points)
        print("\n")
        print("buy and hold walk")
        print(bah_money_walk)

    plt.clf()

    for n in range(len(target_symbols)):
        mw = [m[n] for m in money_walk]
        plt.plot(mw)
    plt.legend(target_symbols)
    plt.savefig(f'{archive_dir}/plots/at_money_walk.png')

    plt.clf()

    for n in range(len(target_symbols)):
        bahw = [m[n] for m in bah_money_walk]
        plt.plot(bahw)
    plt.legend(target_symbols)
    plt.savefig(f'{archive_dir}/plots/hodl_money_walk.png')

    plt.clf()

    for n in range(len(target_symbols)):
        mm = [m[n] for m in money_metric]
        plt.plot(mm)
    plt.legend(target_symbols)
    plt.savefig(f'{archive_dir}/plots/money_metric.png')

    torch.save(net.state_dict(), f'{archive_dir}/weights/network.pth')
        



# %%

client = Client()

#config
limit = 4 * 24 * 365 # 1 year
interval = Client.KLINE_INTERVAL_15MINUTE
window_size = 160
comission = 0.001
epochs = 3000
learning_rate = 0.09


source_symbols = ['IOTABNB']
target_symbols = ['IOTABNB']

train_bot(client, source_symbols, target_symbols, limit, interval, window_size, comission, epochs, learning_rate, archive=True)

source_symbols = ['BNBEUR']
target_symbols = ['BNBEUR']

train_bot(client, source_symbols, target_symbols, limit, interval, window_size, comission, epochs, learning_rate, archive=True)

source_symbols = ['ADAEUR']
target_symbols = ['ADAEUR']

train_bot(client, source_symbols, target_symbols, limit, interval, window_size, comission, epochs, learning_rate, archive=True)

source_symbols = ['IOTABNB']
target_symbols = ['IOTABNB']

train_bot(client, source_symbols, target_symbols, limit, interval, window_size, comission, epochs, learning_rate, archive=True)

source_symbols = ['ADABNB']
target_symbols = ['ADABNB']

train_bot(client, source_symbols, target_symbols, limit, interval, window_size, comission, epochs, learning_rate, archive=True)