
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

class TradeNet(nn.Module):

    def __init__(self, symbols, window_size):
        super(TradeNet, self).__init__()

        self.DEBUG = False
        num_markets = len(symbols)
        self.symbols = symbols
        self.num_markets = num_markets
        self.window_size = window_size

        self.conv = nn.Conv1d(num_markets, num_markets, window_size)
        nn.init.normal_(self.conv.weight, 0, torch.sqrt(torch.tensor(float(window_size))))

        self.pos_weight = nn.Parameter(torch.rand((num_markets,), requires_grad=True))
        nn.init.normal_(self.pos_weight.data, 0, torch.sqrt(torch.tensor(float(window_size))))

    def forward(self, x, start_pos=None):
        # print(f"\nprices: \n\n{x}")

        if self.DEBUG:
            print(f"FORWARDING:")

        if self.DEBUG:
            print(list(self.parameters()))

        start_pos = torch.ones([1,self.num_markets]) * -1 if start_pos == None else start_pos

        current_pos = start_pos #torch.sqrt(torch.tensor(float(self.window_size)))

        if self.DEBUG:
            print(f"X: {x}\n start_pos: {start_pos}\ncurrent_pos: {current_pos}")

        #convolution
        x = self.conv(x)

        if self.DEBUG:
            print(f"conv(x): {x}")


        #split tensor
        x = list(torch.split(x, 1, dim=2))

        if self.DEBUG:
            print(f"splitted x: {x}")

        #compute positions
        for i in range(len(x)):
            if self.DEBUG:
                print(f"x[{i}]: {x[i]}")
            x[i] = torch.reshape(x[i], [1,self.num_markets])
            if self.DEBUG:
                print(f"reshaped x[{i}]: {x[i]}")
            x[i] = torch.tanh((current_pos * self.pos_weight) + x[i]) # multiplikator in tanh reinmachen damit es eher an -1 1 geht?
            if self.DEBUG:
                print(f"activation x[{i}] with pos: {x[i]}")
            current_pos = x[i]
            if self.DEBUG:
                print(f"current_pos: {current_pos}")

        positions = torch.stack([start_pos] + x, 2)

        if self.DEBUG:
            print(f"positions: {positions}")

        return positions

    def train(self, train_prices, validation_prices, comission, epochs, learning_rate):

        optimizer = optim.Adam(self.parameters(), learning_rate)
        #optimizer = optim.RMSprop(self.parameters(), learning_rate)

        money_l = []
        
        for epoch in range(epochs):
            
            positions = self.forward(train_prices.detach().clone())
            returns = self.compute_returns(positions, train_prices, comission)
            loss = self.loss(returns)

            print(f"EPOCH: {epoch} loss: {loss}")

            optimizer.zero_grad()
            loss.backward()#torch.ones([self.num_markets]))
            optimizer.step()

            money_l.append(self.validate(validation_prices, comission))

            if epoch % max(int(epochs / 50), 1) == 0:
                
                money = self.validate(validation_prices, comission, save_plot=False)
                print(f"Made {money} out of 1 for each in {validation_prices.shape[-1]} time steps.")

        money = self.validate(validation_prices, comission, save_plot=False)
        print(f"Made {money} out of 1 in {validation_prices.shape[-1]} time steps.")

        

        for n in range(self.num_markets):
            plt.plot([m[n] for m in money_l])

        plt.suptitle("Money")
        plt.legend(self.symbols)
        plt.savefig("plots/money.png")

   
    def validate(self, validation_prices, comission, initial_money=1, save_plot=False):

        if self.DEBUG:
            print("VALIDATION:")

            print(f"validation_prices: {validation_prices}")

        
        positions = torch.sign(self.forward(validation_prices))

        returns = self.compute_returns(positions, validation_prices, comission)

        if self.DEBUG:
            print(f"validation_prices: {validation_prices} \npositions: {positions} \nreturns: {returns}")

        num_markets = returns.shape[0]

        money = initial_money * torch.ones([returns.shape[0], 1])
        
        for n in range(num_markets):
            for r in returns[n]:
                money[n] *= (1 + r) 

        money = money.cpu().detach().numpy()   

        return money

    def compute_returns(self, positions: torch.FloatTensor, prices: torch.FloatTensor, comission: float):

        assert isinstance(positions, torch.FloatTensor)
        assert isinstance(prices, torch.FloatTensor)
        assert isinstance(comission, float)

        if self.DEBUG:
            print(f"COMPUTING RETURNS:")
            print(f"positions: {positions}\nprices: {prices}")

        diff = prices.shape[-1] - positions.shape[-1]

        pos_t = positions[:,:,:-1][0]
        prices_t = prices[:, :, diff+1:][0]

        if self.DEBUG:
            print(f"pos_t: {pos_t}\nprices_t: {prices_t}")

        comissions = (((pos_t[:,1:] - pos_t[:,:-1]) / 2) ** 2) * comission
        comissions = torch.cat((torch.zeros([comissions.shape[0], 1]), comissions), -1)

        R = (pos_t * prices_t) * (1 - comissions)

        if self.DEBUG:
            print(f"R: {R}")

        return R

    def loss(self, R):

        if self.DEBUG:
            print("LOSS:")
            print(f"R: {R}\nR shape: {R.shape}")

        R_split = list(torch.split(R, 1, dim=0))

        loss = torch.zeros([len(R_split)])
        for n in range(len(R_split)):
            
            curr = R_split[n][0]
            if self.DEBUG:
                print(f"curr: {curr.grad_fn}")
            mean = torch.mean(curr)

            min_squared = curr[torch.nonzero(curr < 0)] ** 2

            if len(min_squared) == 0:
                min_squared = torch.tensor(1.0)

            sortino = mean / torch.sqrt(torch.mean(min_squared))

            

            loss[n] = -sortino
            if self.DEBUG:
                print(f"mean: {mean}\nmin_squared: {min_squared}\nsortino: {sortino}\nsum_loss: {loss}")
                print(f"mean: {mean.grad_fn}\nmin_squared: {min_squared.grad_fn}\nsortino: {sortino.grad_fn}\nsum_loss: {sum_loss.grad_fn}")

        return loss.sum()

    def walk(self, validation_prices, comission, initial_money=1, save_plot=False):

        positions = torch.ones([1,self.num_markets]) * -1   
        money = initial_money * torch.ones([1, self.num_markets])

        money_l = [money]

        trade_points = []

        for n in range(validation_prices.shape[-1] - self.window_size):
            print("-------------------------------")
            window = validation_prices[:,:,n:self.window_size + n]
            new_positions = torch.sign(self.forward(window, positions)[:,:,-1])
            next_prices = validation_prices[:,:,self.window_size + n]

            comissions = torch.abs((positions - new_positions)/2) * comission
            next_returns = (1 + (new_positions * next_prices)) * (1 - comissions)

            next_money = money_l[-1] * next_returns
            money_l.append(next_money)

            if not torch.equal(positions,new_positions):
                print(f"TRADING AT STEP {n}")
                print(f"comissions:\n{comissions}")

                trade_points.append(next_money.clone().detach())
                

            positions = new_positions

            

        if save_plot:
            plt.clf()

            for n in range(self.num_markets):
                plt.plot([m[0][n] for m in money_l])
            plt.scatter(range(len(trade_points)), trade_points)

            plt.suptitle("Money Walk")
            plt.legend(self.symbols)
            plt.savefig("plots/money_walk.png")

            



class TradeNetSlow(nn.Module):

  def __init__(self, returns, comission, window_size):
    super(TradeNet, self).__init__()

    self.returns = returns
    self.window_size = window_size
    self.comission = comission

    #weights for returns
    normal = torch.distributions.Normal(0, torch.sqrt(torch.tensor(window_size, dtype=torch.float64)))
    self.return_weights = normal.sample([window_size])
    self.return_weights.require_grad = True

    #bias
    self.bias = torch.tensor(0.0, requires_grad=True)

    #weight for last position
    self.pos_weight = torch.tensor(0.5, requires_grad=True)

    # make parameters
    self.return_weights = torch.nn.Parameter(self.return_weights)
    self.pos_weight = torch.nn.Parameter(self.pos_weight)
    self.bias = torch.nn.Parameter(self.bias)

    self.register_parameter(name='return weights', param=self.return_weights)
    self.register_parameter(name='bias weight', param=self.bias)
    self.register_parameter(name="position weight", param=self.pos_weight)

  def forward(self, x):
    if x == None:
      x = self.returns
    F = [torch.tensor(0.0) for n in range(len(x))]
    R = [torch.tensor(0.0) for n in range(len(x))]


    for t in range(self.window_size, len(x)):
      F[t] += self.pos_weight * F[t-1]
      F[t] += self.bias
      F[t] += (self.return_weights * torch.tensor(self.returns[t - self.window_size:t])).sum()
      F[t] = torch.tanh(F[t])
      R[t] = F[t-1] * torch.tensor(self.returns[t])
      R[t] *= torch.tensor(1) - (torch.tensor(self.comission) * (((F[t] - F[t-1]) / 2) ** 2))


    
    return F, R

  def train(self, train, validation, epochs, lr):

    optimizer = optim.Adam(net.parameters(), lr=lr)

    for n in range(epochs):

      F, R = self.forward(train)
      sortino = self.loss(R)

      print(f"\nEPOCH {n}\t\tloss: {sortino:.6f}")

      if (n + 1) % 10 == 0:
        print(f"\nMONEY: {self.validate(validation)}")

      optimizer.zero_grad()
      sortino.backward()
      optimizer.step()

    plt.plot([torch.sign(f) for f in F])
    plt.show()


  def validate(self, x):

    positions, returns = self.forward(x)

    money = 1
    for r in returns:
      money *= (1 + r)

    return money.cpu().detach().numpy()

  def loss(self, R):
    R_min_squared = [r ** 2 for r in R if r <= 0]

    size_R = torch.tensor(len(R), dtype=torch.float64)
    size_R_min_squared = torch.tensor(len(R_min_squared), dtype=torch.float64)
    
    sum_R = torch.tensor(0.0)
    for n in range(len(R)):
      sum_R += R[n]
    
    mean_R = sum_R / size_R

    sum_R_min_squared = torch.tensor(0.0)    
    for n in range(len(R_min_squared)):
      sum_R_min_squared += R_min_squared[n]

    mean_R_min_squared = sum_R_min_squared / size_R_min_squared

    sortino = mean_R / torch.sqrt(mean_R_min_squared)

    return -sortino
