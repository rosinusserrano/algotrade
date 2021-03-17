from tradenet import TradeNet
import torch
from torch.distributions.multinomial import Multinomial

import binance_data as bd
from binance.client import Client





torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=10)





client = Client()

net = TradeNet(['LTCBNB'], ['LTCBNB'], 100)

dir = 'BNBEUR-2.5'
net.load_state_dict(torch.load(f'archive/{dir}/weights/network.pth'))

symbol = dir.split('-')[0]

returns = torch.tensor([[[b['return'] for b in bd.get_data(client, Client.KLINE_INTERVAL_15MINUTE, 20000, symbol)]]])

m = net.walk(returns, returns, 0.001)
v = net.validate(returns, returns, 0.001)
print(v)