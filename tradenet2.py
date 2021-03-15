import torch
import torch.nn as nn
import torch.optim as optim

torch.set_default_dtype(torch.float64)

class TradeNet2(nn.Module):

    def __init__(self, sources, targets, window_size=30, num_features=1):

        super(TradeNet2, self).__init__()

        self.N_sources = len(sources)
        self.N_targets = len(targets)

        self.window_size = window_size
        self.num_features = num_features

        # # normalize input
        # self.input_bn = nn.BatchNorm1d(
        #     self.N_sources * self.window_size + self.N_sources)

        # compute features
        self.features = nn.Linear(
            self.N_sources * window_size + self.N_sources, self.N_targets * num_features)

        # # normalize features
        # self.feature_bn = nn.BatchNorm1d(self.num_features * self.N_targets)

        # TODO find good comment
        self.out = nn.Linear(self.num_features *
                             self.N_targets, self.N_targets)

    def forward(self, x):
        
        x = torch.flatten(x, start_dim=1)
        x = self.features(x)
        print(x)
        x = torch.relu(x)
        print(x)
        x = self.out(x)
        print(x)
        x = torch.tanh(x)
        print(x)

        return x


X = torch.tensor([[[-1, 1, 2, 3, 4], [-1, 5, 6, 7, 8]], [[-1, 3, 5, 8, 2], [-1, 4, 1, 7, 7]]]).double()

Y = torch.tensor([[[-1, 1, 2, 3, 4], [-1, 5, 6, 7, 8]]]).double()


net = TradeNet2(['A', 'B'], ['A'], 4, 2)

net(X)

net(Y)