from tradenet import TradeNet
import torch
from torch.distributions.multinomial import Multinomial

net = TradeNet(['BNBBTC', 'BNBETH'], 3)

m = Multinomial(100, torch.ones([1,2,6]))

x = m.sample()

net.conv.bias.data = torch.ones([2])
net.conv.bias.data[1] *= 0.5
net.conv.weight.data = torch.ones([2,2,3])
net.conv.weight.data[1] *= 0.5


print(x)

for p in net.parameters():
    print(p)

print(net.conv(x))