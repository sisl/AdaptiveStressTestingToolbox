"""`Backward Algorithm <https://arxiv.org/abs/1812.03381>`_ from Salimans and Chen."""
import itertools

import numpy as np
from dowel import logger
from garage.tf.algos.ppo import PPO


class BackwardAlgorithm(PPO):
    r"""Backward Algorithm from Salimans and Chen [1]_.

    Parameters
    ----------
    env : :py:class:`ast_toolbox.envs.go_explore_ast_env.GoExploreASTEnv`
        The environment.
    policy : :py:class:`garage.tf.policies.Policy`
        The policy.
    expert_trajectory : array_like[dict]
        The expert trajectory, an array_like where each member represents a timestep in a trajectory.
        The array_like should be 1-D and in chronological order.
        Each member of the array_like is a dictionary with the following keys:
            - state: The simulator state at that timestep (pre-action).
            - reward: The reward at that timestep (post-action).
            - observation: The simulation observation at that timestep (post-action).
            - action: The action taken at that timestep.
    epochs_per_step : int, optional
        Maximum number of epochs to run per step of the trajectory.
    max_epochs : int, optional
        Maximum number of total epochs to run. If not set, defaults to ``epochs_per_step`` times the number of steps
        in the ``expert_trajectory``.
    skip_until_step : int, optional
        Skip training for a certain number of steps at the start, counted backwards from the end of the trajectory.
        For example, if this is set to 3 for an ``expert_trajectory`` of length 10, training will start from step 7.
    max_path_length : int, optional
        Maximum length of a single rollout.
    kwargs :
        Keyword arguments passed to :doc:`garage.tf.algos.PPO <garage:_apidoc/garage.tf.algos>`

    References
    ----------
    .. [1] Salimans, Tim, and Richard Chen. "Learning Montezuma's Revenge from a Single Demonstration."
     arXiv preprint arXiv:1812.03381 (2018). `<https://arxiv.org/abs/1812.03381>`_
    """

    def __init__(self,
                 env,
                 policy,
                 expert_trajectory,
                 epochs_per_step=10,
                 max_epochs=None,
                 skip_until_step=0,
                 max_path_length=500,
                 **kwargs):

        self.max_epochs_per_step = epochs_per_step
        # Input settings related to expert trajectory
        self.max_steps = max_path_length
        self.skip_until_step = skip_until_step
        self.expert_trajectory = expert_trajectory
        self.expert_trajectory_last_step = len(self.expert_trajectory) - 1

        # Get initialization variables
        self.first_iteration_num = np.minimum(self.skip_until_step, self.expert_trajectory_last_step)
        self.first_step_num = np.maximum(0, self.expert_trajectory_last_step - self.first_iteration_num)
        self.num_steps = len(self.expert_trajectory) - self.first_iteration_num

        if max_epochs is None:
            # Set max epochs to the sum of running the max at each step for each step
            self.max_epochs = self.max_epochs_per_step * self.num_steps
        else:
            # Set max epochs to input limit
            self.max_epochs = max_epochs
            if self.max_epochs_per_step * self.num_steps > self.max_epochs:
                # Reduce number of epochs per step if it would violate given max epochs constraint
                self.max_epochs_per_step = np.maximum(1, self.max_epochs // self.num_steps)

        self.env = env
        self.policy = policy

        self.env.set_param_values([None], robustify_state=True, debug=False)

        super(BackwardAlgorithm, self).__init__(policy=policy,
                                                max_path_length=max_path_length,
                                                **kwargs)

    def train(self, runner):
        r"""Obtain samplers and start actual training for each epoch.

        Parameters
        ----------
        runner : :py:class:`garage.experiment.LocalRunner <garage:garage.experiment.LocalRunner>`
            ``LocalRunner`` is passed to give algorithm the access to ``runner.step_epochs()``, which provides services
            such as snapshotting and sampler control.

        Returns
        -------
        full_paths : array_like
            A list of the path data from each epoch.
        """
        max_reward = -np.inf
        max_reward_step = -1
        max_final_reward = -np.inf
        expert_trajectory_reward = np.sum(np.array([step['reward'] for step in self.expert_trajectory]))

        full_paths = []
        runner.train_args.n_epochs = self.max_epochs
        # done = False
        for epoch_itr, epoch_paths in self.get_next_epoch(runner=runner):
            # Modify each rollout to include the expert trajectory data up to the step num (where the agent started)
            for rollout_idx, rollout in enumerate(epoch_paths):

                if rollout['rewards'].shape[0] < self.max_path_length:
                    epoch_paths[rollout_idx]['rewards'] = np.concatenate(
                        (self.env_reward, rollout['rewards']))
                    epoch_paths[rollout_idx]['actions'] = np.concatenate(
                        (self.env_action.reshape((-1, rollout['actions'].shape[1])), rollout['actions']))
                    epoch_paths[rollout_idx]['observations'] = np.concatenate(
                        (self.env_observation.reshape((-1, rollout['observations'].shape[1])), rollout['observations']))

            # Process the modified rollouts and optimize
            last_return = self.train_once(epoch_itr, epoch_paths)
            full_paths.append(last_return)

            # Track reward totals so far
            max_reward_this_step = -np.inf
            for path in last_return['paths']:
                path_reward = np.sum(path['rewards'])
                if path_reward >= max_reward_this_step:
                    max_reward_this_step = path_reward
                if path_reward >= max_reward:
                    max_reward = np.sum(path['rewards'])
                    max_reward_step = self.step_num
                if self.step_num == 0 and path_reward > max_final_reward:
                    max_final_reward = np.sum(path['rewards'])

            # We have beat the expert trajectory from this step, back up or end
            if max_reward_this_step >= expert_trajectory_reward:
                if self.step_num == 0:
                    self.done = True
                else:
                    self.done_with_step = True

        print('Backward Results -- Expert Trajectory Reward: ', expert_trajectory_reward, ' -- Best Reward at step ',
              max_reward_step, ': ', max_reward, " -- Best Final Reward: ", max_final_reward)
        return full_paths

    def train_once(self, itr, paths):
        r"""Perform one step of policy optimization given one batch of samples.

        Parameters
        ----------
        itr : int
            Iteration number.
        paths : list[dict]
            A list of collected paths.

        Returns
        -------
        paths : list[dict]
            A list of processed paths
        """
        paths = self.process_samples(itr, paths)

        self.log_diagnostics(paths)
        logger.log('Optimizing policy...')
        self.optimize_policy(itr, paths)
        return paths

    def get_next_epoch(self, runner):
        r"""Wrapper of garage's :py:meth:`runner.step_epochs()
        <garage:garage.experiment.local_runner.LocalRunner.step_epochs>`
        generator to handle initialization to correct trajectory state

        Parameters
        ----------
        runner : :py:class:`garage.experiment.LocalRunner <garage:garage.experiment.LocalRunner>`
            ``LocalRunner`` is passed to give algorithm the access to ``runner.step_epochs()``, which provides services
            such as snapshotting and sampler control.

        Yields
        -------
        runner.step_itr : int
            The current epoch number.
        runner.obtain_samples(runner.step_itr): list[dict]
            A list of sampled rollouts for the current epoch
        """
        try:
            iteration_num = self.first_iteration_num
            self.step_num = self.first_step_num
            epochs_per_this_step = 0
            self.done = False
            self.set_env_to_expert_trajectory_step()
            self.done_with_step = False

            for epoch_num in itertools.takewhile(lambda x: not self.done, runner.step_epochs()):

                yield runner.step_itr, runner.obtain_samples(runner.step_itr)

                runner.step_itr += 1
                epochs_per_this_step += 1

                if (not self.done and
                        (self.done_with_step or epochs_per_this_step == self.max_epochs_per_step)):
                    if self.step_num == 0:
                        self.done = True
                    else:
                        # Back up the algorithm to the next step of the expert trajectory
                        epochs_per_this_step = 0
                        print('------------ Backward Algorithm: Stepping Back from Step: ', self.step_num, ' to Step: ',
                              np.maximum(0,
                                         self.expert_trajectory_last_step -
                                         np.minimum(iteration_num + 1, self.num_steps - 1)), ' ------------------')
                        iteration_num = np.minimum(iteration_num + 1, self.num_steps - 1)
                        self.step_num = np.maximum(0, self.expert_trajectory_last_step - iteration_num)
                        # print(self.step_num)

                        self.set_env_to_expert_trajectory_step()

                        self.done_with_step = False

        finally:
            # Do any clean-up needed
            pass

    def set_env_to_expert_trajectory_step(self):
        r"""Updates the algorithm to use the data from ``expert_trajectory`` up to the current step.

        """
        self.env_state = self.expert_trajectory[self.step_num]['state']
        self.env_reward = np.array([step['reward'] for step in self.expert_trajectory[:self.step_num]])
        self.env_action = np.array([step['action'] for step in self.expert_trajectory[:self.step_num]])
        self.env_observation = np.array([step['observation'] for step in self.expert_trajectory[:self.step_num]])

        self.env.set_param_values([self.env_state], robustify_state=True, debug=False)
