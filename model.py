import torch
import torch.nn as nn
from torch.distributions import Normal

n_obs = 2
n_h = 20
n_acts = 2
# n_obs = 3
# n_h = 40
# n_acts = 1

class Policy(nn.Module):
    def __init__(self, lr):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(n_obs, n_h),
            nn.Tanh(),
            nn.Linear(n_h, n_h),
            nn.Tanh()
        )

        self.mean = nn.Sequential(
            nn.Linear(n_h, n_acts)
        )

        self.log_std = nn.Sequential(
            nn.Linear(n_h, n_acts)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    
    '''
    Taking a maximizing step on given value w.r.t. model parameters
    '''
    def maximize(self, value):
        self.optimizer.zero_grad()
        (-value).backward()
        self.optimizer.step()

    '''
    Generate Gaussian for given state
    '''
    def dist(self, s):
        s = self.main(s)
        mean = self.mean(s)
        std = torch.exp(self.log_std(s).expand_as(mean))
        dist = Normal(mean, std)
        return dist

    '''
    Sample actions *without* gradients
    '''
    def forward(self, s):
        dist = self.dist(s)
        a = torch.tanh(dist.sample())
        return a
    
    '''
    Get the mean action from the current distribution
    '''
    def mean_action(self, s):
        return self.mean(self.main(s))

    '''
    Sample actions and log probabilities *with* gradients
    '''
    def sample_with_grad(self, s):
        dist = self.dist(s)
        x = dist.rsample()
        a = torch.tanh(x)
        log_p = dist.log_prob(x)
        log_p -= torch.log(1 - torch.pow(a, 2) + 1e-6)
        return a, log_p


class Network:
    def __init__(self, model, lr):
        self.model = model()
        self.target_model = model()
        self.τ = 0.995
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    
    def soft_update_target(self):
        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_((self.τ * target_param.data) + ((1 - self.τ) * param.data))
    
    def target(self, *args):
        with torch.no_grad():
            return self.target_model(*args)

    def minimize(self, value):
        self.optimizer.zero_grad()
        value.backward()
        self.optimizer.step()
    
    def __call__(self, *args):
        return self.model(*args)
    
    def __getattr__(self, k):
        return getattr(self.model, k)


class ActionValue(nn.Module):
    def __init__(self):
        super().__init__()

        self.pre_state = nn.Sequential(
            nn.Linear(n_obs, n_h),
            nn.ELU(),
            nn.Linear(n_h, n_h // 2),
            nn.ELU()
        )

        self.pre_action = nn.Sequential(
            nn.Linear(n_acts, n_h // 2),
            nn.ELU()
        )

        self.main = nn.Sequential(
            nn.Linear(n_h, n_h),
            nn.ELU(),
            nn.Linear(n_h, 1)
        )
    
    def forward(self, s, a):
        s = self.pre_state(s)
        a = self.pre_action(a)
        return self.main(torch.cat([s, a], -1))