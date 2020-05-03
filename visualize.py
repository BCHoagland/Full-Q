import numpy as np
from visdom import Visdom

from env import Env

'''
Print reward map
'''
def print_reward_map():
    x,y = np.mgrid[0:100:1, 0:100:1]
    coords = np.empty(x.shape + (2,))
    coords[:, :, 0] = x; coords[:, :, 1] = y
    coords = np.reshape(coords, (x.shape[0], x.shape[1], 2))
    rewards = np.zeros((x.shape[0], x.shape[1]))

    env = Env()
    for i in range(100):
        for j in range(100):
            rewards[i][j] = env.reward(coords[i][j])

    viz = Visdom()
    viz.heatmap(
        win='map',
        X=rewards.transpose(),
        opts=dict(
            # colormap='Electric',
        )
    )