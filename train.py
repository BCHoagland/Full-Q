import gym
import torch

from model import Network, Policy, ActionValue
from storage import Storage
from env import Env
from map_env import MapEnv
from visualize import plot_reward


###################
# HYPERPARAMETERS #
###################
batch_size = 128
num_timesteps = 1e4
vis_iter = 100
lr = 3e-4
γ = 0.99


###########
# OBJECTS #
###########
env = MapEnv()
# env = Env(gym.make('Pendulum-v0'))
π = Policy(lr)
Q1 = Network(ActionValue, lr)
Q2 = Network(ActionValue, lr)
buffer = Storage()


class Behavior:
    def __call__(self, *args):
        return torch.rand(2) * 2 - 1 
    
    def prob(self, s, a):
        return 1/2

b = Behavior()


########################
# EVALUATE MEAN POLICY #
########################
def eval_policy():
    ep_r = 0
    s = env.reset()
    while True:
        with torch.no_grad():
            a = π.mean_action(s)
            s, r, done = env.step(a)
            ep_r += r
            if done:
                plot_reward(ep_r)
                return


###############################
# GET EXPLORATORY TRANSITIONS #
###############################
s = env.reset()
for _ in range(int(num_timesteps)):

    # interact with environment
    with torch.no_grad():
        a = b(s)
        s2, r, done = env.step(a)

        buffer.store((s, a, r, s2, done))

        if done:
            s = env.reset()
        else:
            s = s2


#######################
# OFF-POLICY TRAINING #
#######################
for i in range(int(num_timesteps)):
    s, a, r, s2, d = buffer.sample(batch_size)
    m = 1 - d

    # improve Q-function
    with torch.no_grad():
        a2 = π(s2)
        y = r + m * γ * torch.min(Q1.target(s2, a2), Q2.target(s2, a2))
    q1_loss = ((Q1(s,a) - y) ** 2).mean()
    q2_loss = ((Q2(s,a) - y) ** 2).mean()
    Q1.minimize(q1_loss)
    Q2.minimize(q2_loss)

    # improve policy
    new_a, log_p = π.sample_with_grad(s)
    with torch.no_grad(): 
        ratio = log_p.exp() / b.prob(s, a)
    q = Q1(s, new_a)
    objective = (ratio * (q.detach() * log_p + q)).mean()
    π.maximize(objective)

    # update target Q function
    Q1.soft_update_target()
    Q2.soft_update_target()

    # plot progress occasionally
    if i % vis_iter == vis_iter - 1: eval_policy()