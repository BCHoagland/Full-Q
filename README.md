# Full-Q
Experiments with using policy re-parameterization to allow for the full computation of the Q-learning gradient when using entirely off-policy learning.

[Degris et al., 2013] derive in the discrete state/action case:

![\nabla_\theta J(\theta) = \sum_s \mu_b(s) \sum_a \big\[ \nabla_\theta \pi(a|s) Q^\pi(s,a) + \pi(a|s) \nabla_\theta Q^\pi(s,a) \big\]](https://render.githubusercontent.com/render/math?math=%5Cnabla_%5Ctheta%20J(%5Ctheta)%20%3D%20%5Csum_s%20%5Cmu_b(s)%20%5Csum_a%20%5Cbig%5B%20%5Cnabla_%5Ctheta%20%5Cpi(a%7Cs)%20Q%5E%5Cpi(s%2Ca)%20%2B%20%5Cpi(a%7Cs)%20%5Cnabla_%5Ctheta%20Q%5E%5Cpi(s%2Ca)%20%5Cbig%5D)

then drop the last term as an approximation since it's hard to calculate. If the action space becomes continuous, then we can use the re-parameterization trick to allow gradients to flow through this last term. This lets us calculate the entire gradient.
