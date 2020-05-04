import torch

from model import Network, Policy, ActionValue
from storage import Storage
from env import Env
from visualize import eval_map, plot_reward, plot_states


batch_size = 128
num_timesteps = 1e5
lr = 1e-4
γ = 0.99


env = Env()
π = Policy(lr)
QV = Network(ActionValue, lr)
# V = Network(Value, lr)
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

    # improve Q-function and V-function
    with torch.no_grad():
        a2 = π(s2)
        y = r + m * γ * QV.target_model.q(s2, a2)
    q_loss = ((QV.q(s,a) - y) ** 2).mean()
    v_loss = ((QV.v(s) - y) ** 2).mean()
    QV.minimize(q_loss + v_loss)

    # improve policy
    new_a, log_p = π.sample_with_grad(s)                                #! make sure log_prob is calculated correctly
    with torch.no_grad(): 
        ratio = log_p.exp() / 4                                           #! add in behavioral policy
    adv = QV.adv(s, a)
    objective = (ratio * (adv.detach() * log_p + adv)).mean()
    π.maximize(objective)

    # update target Q function
    QV.soft_update_target()



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
                # ['V(s)', 'Q(s, [0,0])'],
                # [QV.v, lambda x: QV.q(x, torch.zeros(2))]
                ['Q(s, [0,0])'],
                [lambda x: QV.q(x, torch.zeros(2))]
            )
            eval_policy()
            del ep_s[:]
            s = env.reset()
        else:
            s = s2
    
    # update networks
    update()
