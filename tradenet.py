
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt


class TradeNet(nn.Module):

    def __init__(self, source_symbols, target_symbols, window_size):
        super(TradeNet, self).__init__()

        self.DEBUG = False

        num_sources = len(source_symbols)
        self.source_symbols = source_symbols
        self.num_sources = num_sources

        num_targets = len(target_symbols)
        self.target_symbols = target_symbols
        self.num_targets = num_targets

        self.window_size = window_size

        self.conv = nn.Conv1d(num_sources, num_targets, window_size)
        nn.init.normal_(self.conv.weight, 0, torch.sqrt(
            torch.tensor(float(window_size))))

        self.pos_weight = nn.Parameter(torch.rand(
            (num_targets, 1), requires_grad=True))
        nn.init.normal_(self.pos_weight.data, 0, torch.sqrt(
            torch.tensor(float(window_size))))

    def forward(self, x, start_pos=None):
        # print(f"\nprices: \n\n{x}")

        if self.DEBUG:
            print(f"FORWARDING:")

        if self.DEBUG:
            print(list(self.parameters()))

        start_pos = torch.ones([self.num_targets, 1]) * - \
            1 if start_pos == None else start_pos

        # torch.sqrt(torch.tensor(float(self.window_size)))
        current_pos = start_pos

        if self.DEBUG:
            print(f"X: {x}\n start_pos: {start_pos}\ncurrent_pos: {current_pos}")

        # convolution
        x = self.conv(x)

        if self.DEBUG:
            print(f"conv(x): {x}")

        # split tensor
        x = list(torch.split(x, 1, dim=2))

        if self.DEBUG:
            print(f"splitted x: {x}")

        # compute positions
        for i in range(len(x)):
            if self.DEBUG:
                print(f"x[{i}]: {x[i]}")
            x[i] = torch.reshape(x[i], [self.num_targets, 1])
            if self.DEBUG:
                print(f"reshaped x[{i}]: {x[i]}")
            # TODO multiplikator in tanh reinmachen damit es eher an -1 1 geht?
            x[i] = torch.tanh(((current_pos * self.pos_weight) + x[i]))
            if self.DEBUG:
                print(f"activation x[{i}] with pos: {x[i]}")
            current_pos = torch.sign(x[i])
            if self.DEBUG:
                print(f"current_pos: {current_pos}")

        positions = torch.cat([start_pos] + x, dim=1)
        positions = torch.unsqueeze(positions, dim=0)

        if self.DEBUG:
            print(f"positions: {positions}")

        return positions

    def train(self, source_train_prices, target_train_prices, source_validation_prices, target_validation_prices, comission, epochs, learning_rate):

        optimizer = optim.Adam(self.parameters(), learning_rate)
        #optimizer = optim.RMSprop(self.parameters(), learning_rate)

        money_l = []

        for epoch in range(epochs):

            positions = self.forward(source_train_prices.type(torch.float64))
            returns = self.compute_returns(positions, target_train_prices, comission)
            loss = self.loss(returns)

            print(f"EPOCH: {epoch} loss: {loss}")

            optimizer.zero_grad()
            loss.backward(torch.ones_like(loss.clone().detach()))
            optimizer.step()

            money_l.append(self.validate(
                source_validation_prices, target_validation_prices, comission).double())

            # if torch.equal(first_param, list(self.parameters())[0].data):
            #     print(list(self.parameters()))

            if epoch % max(int(epochs / 50), 1) == 0:

                money = self.validate(source_validation_prices, target_validation_prices, comission)
                print(
                    f"Made {money} out of 1 for each in {source_validation_prices.shape[-1] - self.window_size} time steps.")

        money = self.validate(source_validation_prices, target_validation_prices, comission)
        print(
            f"Made {money} out of 1 in {source_validation_prices.shape[-1] - self.window_size} time steps.")

        return money_l

    def validate(self, source_validation_prices, target_validation_prices, comission):

        if self.DEBUG:
            print("VALIDATION:")

            print(f"validation_prices: {source_validation_prices}")

        positions = torch.sign(self.forward(source_validation_prices))

        returns = self.compute_returns(
            positions, target_validation_prices, comission)[:, 1:]

        if self.DEBUG:
            print(
                f"validation_prices: {source_validation_prices} \npositions: {positions} \nreturns: {returns}")

        money = torch.ones([returns.shape[0], 1])

        for n in range(self.num_targets):
            for r in returns[n]:
                money[n] *= r

        money = money.cpu().detach()

        return money

    def compute_returns(self, positions: torch.DoubleTensor, prices: torch.DoubleTensor, comission: float):

        assert isinstance(positions, torch.DoubleTensor)
        assert positions.shape[0] == 1, f"positions got shape {positions.shape}"
        assert positions.shape[
            1] == self.num_targets, f"positions got shape {positions.shape} and there are {self.num_targets} markets"
        assert isinstance(prices, torch.DoubleTensor)
        assert prices.shape[0] == 1
        assert prices.shape[1] == self.num_targets
        assert isinstance(comission, float)

        diff = prices.shape[-1] - positions.shape[-1]

        if self.DEBUG:
            print(f"COMPUTING RETURNS:")
            print(f"positions: {positions}\nprices: {prices}")
            print(f"diff: {diff}")

        pos_t = positions[:, :, :-1][0]
        prices_t = prices[:, :, diff+1:][0]

        comissions = (((pos_t[:, 1:] - pos_t[:, :-1]) / torch.tensor(2.))
                      ** torch.tensor(2.)) * torch.tensor(comission)
        comissions = torch.cat(
            (torch.zeros([comissions.shape[0], 1]), comissions), -1)

        if self.DEBUG:
            print(
                f"pos_t: {pos_t}\nprices_t: {prices_t}\ncomissions: {comissions}")

        R = (torch.tensor(1.) + (pos_t * prices_t)) * \
            (torch.tensor(1.) - comissions)

        if self.DEBUG:
            print(f"R: {R}")

        return R

    def loss(self, R):

        if self.DEBUG:
            print("LOSS:")
            print(f"R: {R}\nR shape: {R.shape}")

        R = R - torch.tensor(1.)

        R_split = list(torch.split(R, 1, dim=0))

        loss = torch.zeros([len(R_split)])
        for n in range(len(R_split)):

            curr = R_split[n][0]
            if self.DEBUG:
                print(f"curr: {curr.grad_fn}")
            mean = torch.mean(curr)

            min_squared = curr[torch.nonzero(curr < 0)] ** torch.tensor(2.)

            if len(min_squared) == 0:
                min_squared = torch.tensor(1.0)

            sortino = mean / torch.sqrt(torch.mean(min_squared))

            loss[n] = -sortino
            if self.DEBUG:
                print(
                    f"mean: {mean}\nmin_squared: {min_squared}\nsortino: {sortino}\nsum_loss: {loss}")
                print(
                    f"mean: {mean.grad_fn}\nmin_squared: {min_squared.grad_fn}\nsortino: {sortino.grad_fn}\nsum_loss: {loss.grad_fn}")

        return loss

    def walk(self, source_validation_prices, target_validation_prices, comission):

        positions = torch.ones([self.num_targets, 1]) * -1
        positions = positions.double()
        money = torch.ones([self.num_targets, 1])

        money_l = [money]

        trade_points = []

        for n in range(source_validation_prices.shape[-1] - self.window_size):
            window = source_validation_prices[:, :, n:self.window_size + n]
            assert window.shape[-1] == self.window_size
            new_positions = torch.sign(self.forward(window, positions))
            new_positions = new_positions[0, :, -1].clone().detach().reshape([self.num_targets, 1])
            next_prices = target_validation_prices[:, :,
                                            self.window_size + n].reshape([self.num_targets, 1])

            comissions = torch.abs((positions - new_positions)/2) * comission
            next_returns = (1 + (new_positions * next_prices)
                            ) * (torch.tensor(1) - comissions)

            next_money = money_l[-1] * next_returns
            money_l.append(next_money)

            if not torch.equal(positions, new_positions):
                # print(f"TRADING AT STEP {n} for:")
                # print([self.symbols[i] for i in ((positions - new_positions) != 0).nonzero(as_tuple=True)[0]])
                if self.DEBUG or False:
                    print(f"window: {window}")
                    print(f"new_pos: \n{new_positions}")
                    print(f"comissions:\n{comissions}")
                    print(f"next_prices: \n{next_prices}")
                    print(f"next returns \n{next_returns}")

                trade_points.append([(positions - new_positions) != 0., n])

            positions = new_positions

        print(f"Made {next_money} at end of walk")

        return money_l, trade_points
