#%%
from binance.client import Client

import matplotlib.pyplot as plt

import torch
import torch.optim as optim

import binance_data as bd
from tradenet import TradeNet

from torchviz import make_dot

#%%


client = Client()

#congif
symbols = ['BNBBTC', 'ETHBTC']
limit = 4500
interval = Client.KLINE_INTERVAL_30MINUTE
window_size = 70
comission = 0.001
epochs = 1200 #min 50
learning_rate = 0.1

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
prices = torch.stack(currencies).reshape([1, len(symbols), length])
# prices = BNBBTC.reshape([1, 1, length])


#%%
net = TradeNet(symbols, window_size)

train, validation = prices[:,:,:length80], prices[:,:,length80-window_size:]

b = net(validation)

#make_dot(b, dict(net.named_parameters())).render("rnn_torchviz", format="png")

print("------------------")
print("\n\n\nTRAIN:")
net.train(train, validation, comission, epochs, learning_rate)
print("------------------")
print("\n\n\nWALK:")
net.walk(validation, comission, initial_money=1,save_plot=True)

# %%
