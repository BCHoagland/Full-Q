import torch

from model import Network, Policy, ActionValue, Value
from storage import Storage
from env import Env
from visualize import eval_map, plot_reward, plot_states


batch_size = 128
num_timesteps = 1e5
lr = 3e-4
γ = 0.99


env = Env()
π = Policy(lr)
Q = Network(ActionValue, lr)
V = Network(Value, lr)
buffer = Storage()


########################
# EVALUATE MEAN POLICY #
########################
def eval_policy():
    states = []
    ep_r = 0
    s = env.reset()
    while True:
        with torch.no_grad():
            a = π.mean_action(s)
            s, r, done = env.step(a)
            ep_r += r
            states.append(s.tolist())
            if done:
                plot_reward(ep_r)
                plot_states(states, 'eval')
                return


###############
# UPDATE RULE #
###############
def update():
    s, a, r, s2, d = buffer.sample(batch_size)
    m = 1 - d

    # improve Q-function
    with torch.no_grad():
        a2 = π(s2)
        y = r + m * γ * Q.target(s2, a2)
    q_loss = ((Q(s,a) - y) ** 2).mean()
    Q.minimize(q_loss)

    # improve V-function
    with torch.no_grad():
        y = r + m * γ * V.target(s2)
    v_loss = ((V(s) - y) ** 2).mean()
    V.minimize(v_loss)

    # improve policy
    new_a, log_p = π.sample_with_grad(s)                                #! make sure log_prob is calculated correctly
    with torch.no_grad(): 
        ratio = log_p.exp() / 4                                           #! add in behavioral policy
    adv = Q(s, new_a)
    objective = (ratio * (adv.detach() * log_p + adv)).mean()
    π.maximize(objective)

    # update target Q function
    Q.soft_update_target()
    V.soft_update_target()



ep_s = []
eval_map(['True Reward'], [env.reward])

s = env.reset()
for _ in range(int(num_timesteps)):

    # interact with environment
    with torch.no_grad():
        a = torch.rand(2) * 2 - 1                               #! turn into actual object
        s2, r, done = env.step(a)
        ep_s.append(s2.tolist())

        buffer.store((s, a, r, s2, done))

        if done:
            plot_states(ep_s)
            eval_map(
                ['V(s)', 'Q(s, [0,0])'],
                [V, lambda x: Q(x, torch.zeros(2))]
            )
            eval_policy()
            del ep_s[:]
            s = env.reset()
        else:
            s = s2
    
    # update networks
    update()
