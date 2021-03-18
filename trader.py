import sys
import os
import json
from datetime import datetime

import torch
from binance.client import Client

from tradenet import TradeNet
import binance_data as bd

# environejnfe
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=10)

# config
bot_dir = 'LTCBNB-2.2'
symbol = bot_dir.split('-')[0]
verify = False
client = Client()
dir_path = os.path.dirname(os.path.realpath(__file__))

### cli arguments
# first arg is filename
skip = True
for i in range(len(sys.argv)-1):
    key = sys.argv[i]
    val = sys.argv[i+1]
    if skip:
        skip = False
        continue
    assert key[0:2] == '--' , 'Arguments have to start with a double dash "--"' 
    param = key[2:]
    if param == 'dir':
        bot_dir = val
        symbol = bot_dir.split('-')[0]
        skip = True
    elif param == 'verify':
        verify = int(val)
        skip = True

info_f = open(f"{dir_path}/archive/{bot_dir}/info.json")
info = json.load(info_f)

window_size = info['window_size']
comission = info['comission']
interval = info['interval']


# position ahdnling
def load_position():
    #check if money file exists, otherwise create it
    if not os.path.isfile(f'{dir_path}/{bot_dir}.money'):
        f = open(f'{dir_path}/{bot_dir}.money', "w+")
        print("creating money file...")
        content = "1 base"
        f.write(content)
        f.close()
    else:
        content = open(f'{dir_path}/{bot_dir}.money').readline()
    pos = content.split(' ')[1]
    pos = 1 if pos == 'quote' else -1
    return pos

# get net
net = TradeNet([symbol], [symbol], window_size)
net.load_state_dict(torch.load(f'{dir_path}/archive/{bot_dir}/weights/network.pth'))




### IF ONLY VERIFYING
if verify != False:
    returns = torch.tensor([[[val['return'] for val in bd.get_data(client, interval, window_size + verify, symbol)]]])
    val = net.validate(returns, returns, comission)
    print(f"Made {val.detach().numpy()} money in {verify} steps with interval size of {interval}")
    exit()




### ACTUAL BOT
# next position
position = torch.tensor(load_position()).reshape([1,1])
returns = torch.tensor([[[val['return'] for val in bd.get_data(client, interval, window_size, symbol)]]])
next_pos = int(net(returns, position).sign().squeeze()[1].clone().detach().numpy())

if next_pos != position:
    # if trading
    money_file = open(f'{dir_path}/{bot_dir}.money')
    money = money_file.readline().split(' ')[0]
    print(money)
    money = float(money)
    print(f"money {money}")
    money_file.close()

    price = float(client.get_avg_price(symbol=symbol)['price'])
    print(f"price {price}")

    if next_pos == -1:
        c = 'base'
        price = 1 / price
    else:
        c = 'quote'

    new_money = (money / price) * 0.999

    money_file = open(f'{dir_path}/{bot_dir}.money', "w")
    money_file.write(f'{new_money} {c}')
    money_file.close()




