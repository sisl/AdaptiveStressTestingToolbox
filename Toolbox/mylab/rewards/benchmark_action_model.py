
# useful packages for math and debugging
from mylab.rewards.action_model import ActionModel

import numpy as np
import scipy.spatial as spatial

def inv_cumsum(arr, axis = 0):
    s1 = tuple(slice(1,None) if i == axis else slice(None, None) for i in range(arr.ndim))
    s2 = tuple(slice(None,-1) if i == axis else slice(None, None) for i in range(arr.ndim))
    cp = arr.copy()
    cp[s1] -= cp[s2]
    return cp

def a_to_x(arr, axis=0): return np.cumsum(np.cumsum(arr, axis=axis), axis=axis)
def x_to_a(arr, axis=0): return inv_cumsum(inv_cumsum(arr, axis=axis), axis=axis)

def samp_traj(dim = 2, T = 50):
    return a_to_x(np.random.normal(size=(T,dim)))

def traj_md(traj):
    if traj.shape[0] == 0:
        return 0
    a = x_to_a(traj)
    mu = np.zeros_like(a[0,:])
    cov = np.eye(mu.shape[0])
    return -np.sum([spatial.distance.mahalanobis(mu, a[i, :], cov) for i in range(a.shape[0])])

def traj_dist(traj1, traj2, lam = 20):
    T = min(traj1.shape[0], traj2.shape[0])
    diff_len = abs(traj1.shape[0] - traj2.shape[0])
    traj_diff = np.mean(np.linalg.norm(traj1[:T, :] - traj2[:T, :], axis=0))
    return traj_diff + lam*diff_len


# Define the class, inherit from the base
class BenchmarkActionModel(ActionModel):
    def __init__(self):
        super().__init__()

    def log_prob(self, action, **kwargs):
        """
        returns the log probability of an action
        Input
        -----
        action : the action to get the log probabilty of
        Outputs
        -------
        logp [Float] : log probability of the action
        """
        info = kwargs['info']
        action_sequence = info['action_sequence']
        assert((action_sequence[-1] == action).all())
        if action_sequence.shape[0] == 1:
            return traj_md(action_sequence)
        else:
            return traj_md(action_sequence) - traj_md(action_sequence[:-1])
