#%%
# BINANCE
from binance.client import Client

# UTILS
import matplotlib.pyplot as plt
from pathlib import Path
from uuid import uuid4

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

def train_bot(client, symbols, limit, interval, window_size, comission, epochs, learning_rate, archive=False):
    
    if archive:
        # setup archive directory
        ## create random ID
        archive_dir = "archive/" + str(uuid4())
        ## create directory
        Path(archive_dir).mkdir(parents=True, exist_ok=True)
        print(archive_dir)
        ## add info file
        # TODO

    #get data from symbols
    currencies = []
    for symbol in symbols:
        print(f"getting symbol {symbol}")
        returns = torch.tensor([b['return'] for b in bd.get_data(client, interval, limit, symbol)])
        print(f"got {len(returns)} from {symbol}")
        currencies.append(returns)

    #get minimum length of data to cut them in euqal lengths
    length = min([len(data) for data in currencies])
    length80 = int(length * 0.8)    #save 80% of length for train val split
    currencies = [returns[-length:] for returns in currencies]

    #make tensor from data
    prices = torch.stack(currencies).reshape([1, len(symbols), length]).double()
    print(prices.dtype)
    # prices = BNBBTC.reshape([1, 1, length])


    #%%
    net = TradeNet(symbols, window_size)

    net.double()

    train, validation = prices[:,:,:length80], prices[:,:,length80-window_size:]

    ref_val = validation.clone().detach()

    assert torch.equal(validation, ref_val)

    print(list(net.parameters()))

    print("\n\n\nTRAIN...")
    money_metric = net.train(train, validation, comission, epochs, learning_rate)

    print(list(net.parameters()))

    assert torch.equal(validation, ref_val)

    print("\n\n\nVALIDATE...")
    print(net.validate(validation, comission))

    print("\n\n\nWALK...")
    money_walk, trade_points = net.walk(validation, comission)

    assert torch.equal(validation, ref_val)

    print("\n\n\nBUY AND HOLD...")
    bah_money_walk = buy_and_hold(validation[:,:,window_size:])

    assert torch.equal(validation, ref_val)

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

    for n in range(len(symbols)):
        mw = [m[n] for m in money_walk]
        plt.plot(mw)
    plt.legend(symbols)
    plt.savefig('plots/at_money_walk.png')

    plt.clf()

    for n in range(len(symbols)):
        bahw = [m[n] for m in bah_money_walk]
        plt.plot(bahw)
    plt.legend(symbols)
    plt.savefig('plots/hodl_money_walk.png')
        



# %%

client = Client()

#config
symbols = ['BNBBTC', 'BNBEUR', 'BTCEUR']
limit = 10000
interval = Client.KLINE_INTERVAL_15MINUTE
window_size = 100
comission = 0.001
epochs = 1000
learning_rate = 0.1

train_bot(client, symbols, limit, interval, window_size, comission, epochs, learning_rate, archive=False)

