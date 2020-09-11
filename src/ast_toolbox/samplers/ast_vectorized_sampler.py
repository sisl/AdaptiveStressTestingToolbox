import numpy as np
from garage.sampler.on_policy_vectorized_sampler import OnPolicyVectorizedSampler

from ast_toolbox.rewards import ExampleAVReward
from ast_toolbox.simulators import ExampleAVSimulator


class ASTVectorizedSampler(OnPolicyVectorizedSampler):
    """A vectorized sampler for AST to handle open-loop simulators.

    Garage usually genearates samples in a closed-loop process. This version of the vectorized sampler instead
    grabs dummy data until the full rollout specification is generated, then goes back and runs the `simulate`
    function to actually obtain results. Rewards are then calculated and the path data is corrected.

    Parameters
    ----------
    algo : :py:class:`garage.np.algos.base.RLAlgorithm`
        The algorithm.
    env : :py:class:`ast_toolbox.envs.ASTEnv`
        The environment.
    n_envs : int
        Number of parallel environments to run.
    open_loop : bool
        True if the simulation is open-loop, meaning that AST must generate all actions ahead of time, instead
        of being able to output an action in sync with the simulator, getting an observation back before
        the next action is generated. False to get interactive control, which requires that `blackbox_sim_state`
        is also False.
    sim : :py:class:`ast_toolbox.simulators.ASTSimulator`
        The simulator wrapper, inheriting from `ast_toolbox.simulators.ASTSimulator`.
    reward_function : :py:class:`ast_toolbox.rewards.ASTReward`
        The reward function, inheriting from `ast_toolbox.rewards.ASTReward`.

    """

    def __init__(self, algo, env, n_envs=1, open_loop=True, sim=ExampleAVSimulator(), reward_function=ExampleAVReward()):

        # pdb.set_trace()
        self.open_loop = open_loop
        self.sim = sim
        self.reward_function = reward_function
        super().__init__(algo, env, n_envs)

    def obtain_samples(self, itr, batch_size=None, whole_paths=False):
        """Sample the policy for new trajectories.

        Parameters
        ----------
        itr : int
            Iteration number.
        batch_size : int
            Number of samples to be collected. If None,
            it will be default [algo.max_path_length * n_envs].
        whole_paths : bool
            Whether return all the paths or not. True
            by default. It's possible for the paths to have total actual
            sample size larger than batch_size, and will be truncated if
            this flag is true.

        Returns
        -------
        : list[dict]
            A list of sampled rollout paths.
            Each rollout path is a dictionary with the following keys:
                - observations (numpy.ndarray)
                - actions (numpy.ndarray)
                - rewards (numpy.ndarray)
                - agent_infos (dict)
                - env_infos (dict)

        """
        # pdb.set_trace()
        paths = super().obtain_samples(itr, batch_size)
        # pdb.set_trace()
        if self.open_loop:
            for path in paths:
                s_0 = path["observations"][0]

                # actions = path['env_infos']['info']['actions']
                actions = path['actions']
                # pdb.set_trace()
                end_idx, info = self.sim.simulate(actions=actions, s_0=s_0)
                # print('----- Back from simulate: ', end_idx)
                if end_idx >= 0:
                    # pdb.set_trace()
                    self.slice_dict(path, end_idx)
                rewards = self.reward_function.give_reward(
                    action=actions[end_idx],
                    info=self.sim.get_reward_info()
                )
                # print('----- Back from rewards: ', rewards)
                # pdb.set_trace()
                path["rewards"][end_idx] = rewards
                # info[:, -1] = path["rewards"][:info.shape[0]]
                # path['env_infos']['sim_info'] = info
                path['env_infos']['sim_info'] = np.zeros_like(path["rewards"])
                # import pdb; pdb.set_trace()

        return paths

    def slice_dict(self, in_dict, slice_idx):
        """Helper function to recursively parse through a dictionary of dictionaries and arrays to slice \
        the arrays at a certain index.

        Parameters
        ----------
        in_dict : dict
            Dictionary where the values are arrays or other dictionaries that follow this stipulation.
        slice_idx : int
            Index to slice each array at.

        Returns
        -------
        dict
            Dictionary where arrays at every level are sliced.

        """
        for key, value in in_dict.items():
            # pdb.set_trace()
            if isinstance(value, dict):
                in_dict[key] = self.slice_dict(value, slice_idx)
            else:
                in_dict[key][slice_idx + 1:, ...] = np.zeros_like(value[slice_idx + 1:, ...])

        return in_dict
