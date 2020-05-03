import numpy as np

class Env:
    def __init__(self):
        self.goal = np.array([100]*2)
        self.mid_goal = self.goal * np.array([1/2, 1/6])
        self.state = np.zeros(2)
        self.t = 0
    
    '''
    Action should be [distance to move in x direction, distance to move in y direction]
    Left/down is negative, right/up is positive
    '''
    def step(self, a):
        self.state += a
        self.t += 1

        return np.copy(self.state), self.reward(self.state), self.done()

    '''
    Reward is [max_r - (distance from (5n/6, n/6))]
    '''
    def reward(self, s):
        dist_from_mid = np.linalg.norm(s - self.mid_goal)
        dist_from_end = np.linalg.norm(s - self.goal)
        return 80 - (0.7 * dist_from_end) - (0.3 * dist_from_mid)               #! remove the '80' later to make using a baseline more important

    '''
    Environment ends after 100 timesteps
    '''
    def done(self):
        return self.t >= 100