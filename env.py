import torch

class Env:
    def __init__(self, gym_env):
        self.env = gym_env
    
    def reset(self):
        return torch.FloatTensor([self.env.reset()]).squeeze()
    
    def step(self, a):
        s2, r, done, _ = self.env.step(a.numpy())
        done = torch.ones(1) if done else torch.zeros(1)
        return torch.FloatTensor([s2]).squeeze(), torch.FloatTensor([r]).squeeze(), done.squeeze()