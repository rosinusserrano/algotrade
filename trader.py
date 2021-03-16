import sys
import os

import torch
from binance.client import Client

from tradenet3 import TradeNet3
import binance_data as bd

# environejnfe
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=10)

# config
market = 'LTC'
curr = 'BNB'
symbol = market + curr
client = Client()
dir_path = os.path.dirname(os.path.realpath(__file__))

# position ahdnling
def load_position():
    pos = int(open(f'{dir_path}/{symbol}.pos').read())
    return pos

def write_position(position):
    open(f'{dir_path}/{symbol}.pos', 'w').write(str(position))

# get net
net = TradeNet3([symbol], [symbol], 100)
net.load_state_dict(torch.load(f'{dir_path}/archive/LTCBNB-2.2/weights/network.pth'))

# next position
position = torch.tensor(load_position()).reshape([1,1])
returns = torch.tensor([[[b['return'] for b in bd.get_data(client, Client.KLINE_INTERVAL_15MINUTE, 100, symbol)]]])
next_pos = int(net(returns, position).sign().squeeze()[1].clone().detach().numpy())
write_position(next_pos)


if next_pos != position:

    money_file = open(f'{dir_path}/{symbol}.money')
    money = money_file.readline().split(' ')[0]
    print(money)
    money = float(money)
    print(f"money {money}")
    money_file.close()

    price = float(client.get_avg_price(symbol=symbol)['price'])
    print(f"price {price}")
    
    if next_pos == -1:
        c = curr
        price = 1 / price
    else:
        c = market

    new_money = (money / price) * 0.999

    money_file = open(f'{dir_path}/{symbol}.money', "w")
    money_file.write(f'{new_money} {c}')
    money_file.close()




