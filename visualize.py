import torch
import numpy as np
from visdom import Visdom


viz = Visdom()


rewards = []
def plot_reward(r):
    rewards.append(r)
    viz.line(
        win='rewards',
        X=np.arange(1,len(rewards)+1),
        Y=np.array(rewards),
        opts=dict(
            title='Episodic Reward for Mean Policy'
        )
    )


# states = []
def plot_states(s, name='data'):
    # states.append(s)
    viz.scatter(
        win='states',
        X=np.array(s),
        name=name,
        update='replace',
        opts=dict(
            markersize=5,
            # xtickmin=-50,
            # xtickmax=100,
            # ytickmin=-50,
            # ytickmax=100,
        )
    )


def eval_map(titles, fns):
    x,y = np.mgrid[-20:100:5, -20:100:5]
    coords = np.empty(x.shape + (2,))
    coords[:, :, 0] = x; coords[:, :, 1] = y
    coords = torch.FloatTensor(np.reshape(coords, (x.shape[0], x.shape[1], 2)))
    rewards = np.zeros((x.shape[0], x.shape[1]))

    for title, fn in zip(titles, fns):
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                rewards[i][j] = fn(coords[i][j]).numpy()

        axis_labels = [str(n) for n in np.arange(-20, 100, 5)]
        viz.heatmap(
            win=title,
            X=rewards.transpose(),
            opts=dict(
                title=title,
                rownames=axis_labels,
                columnnames=axis_labels
            )
        )