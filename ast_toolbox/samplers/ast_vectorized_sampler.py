from garage.sampler.on_policy_vectorized_sampler import OnPolicyVectorizedSampler
import pickle

import tensorflow as tf
import numpy as np

import pdb
from ast_toolbox.simulators.example_av_simulator import ExampleAVSimulator
from ast_toolbox.rewards.example_av_reward import ExampleAVReward

class ASTVectorizedSampler(OnPolicyVectorizedSampler):
    def __init__(self, algo, env, n_envs = 1, open_loop = True, sim = ExampleAVSimulator(), reward_function = ExampleAVReward()):
        # pdb.set_trace()
        self.open_loop = open_loop
        self.sim = sim
        self.reward_function = reward_function
        super().__init__(algo, env, n_envs)

    def obtain_samples(self, itr, batch_size=None, whole_paths=False):
        # pdb.set_trace()
        paths = super().obtain_samples(itr)
        # pdb.set_trace()
        if self.open_loop:
            for path in paths:
                s_0 = path["observations"][0]

                # actions = path['env_infos']['info']['actions']
                actions = path['actions']
                # pdb.set_trace()
                end_idx, info = self.sim.simulate(actions = actions, s_0 = s_0)
                if end_idx >= 0:
                    # pdb.set_trace()
                    self.slice_dict(path, end_idx)
                rewards = self.reward_function.give_reward(
                    action = actions[end_idx],
                    info = self.sim.get_reward_info()
                )
                # pdb.set_trace()
                path["rewards"][end_idx] = rewards
                info[:,-1] = path["rewards"][:info.shape[0]]
                path['env_infos']['cache'] = info

        return paths

    def slice_dict(self, in_dict, slice_idx):
        for key, value in in_dict.items():
            # pdb.set_trace()
            if type(value) is dict:
                in_dict[key] = self.slice_dict(value, slice_idx)
            else:
                in_dict[key][slice_idx + 1:, ...] = np.zeros_like(value[slice_idx + 1:, ...])

        return in_dict
