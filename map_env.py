import torch

class MapEnv:
    def __init__(self):
        self.goal = torch.FloatTensor([100]*2)
        self.mid_goal = self.goal * torch.FloatTensor([1/2, 1/6])


    def reset(self):
        with torch.no_grad():
            self.state = torch.zeros(2)
            self.t = 0

            return torch.clone(self.state)


    '''
    Action should be [distance to move in x direction, distance to move in y direction]
    Left/down is negative, right/up is positive
    '''
    def step(self, a):
        with torch.no_grad():
            # self.state = torch.clamp(self.state + a*5, 0, 100)
            self.state = self.state + a*5
            self.t += 1

            return torch.clone(self.state), self.reward(self.state), self.done()


    '''
    Reward is [max_r - (distance from (5n/6, n/6))]
    '''
    def reward(self, s):
        with torch.no_grad():
            dist_from_mid = torch.norm(s - self.mid_goal, 2)
            dist_from_end = torch.norm(s - self.goal, 2)
            return 70 - (0.7 * dist_from_end) - (0.3 * dist_from_mid)               #! remove the '80' later to make using a baseline more important


    '''
    Environment ends after 100 timesteps
    '''
    def done(self):
        if self.t >= 100: return torch.ones(1)
        return torch.zeros(1)