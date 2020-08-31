import itertools

import numpy as np
from dowel import logger
from garage.tf.algos.ppo import PPO


class BackwardAlgorithm(PPO):
    """Backward Algorithm from Salimans and Chen.

    See .

    Args:
        env_spec (garage.envs.EnvSpec): Environment specification.
        policy (garage.tf.policies.base.Policy): Policy.
        baseline (garage.tf.baselines.Baseline): The baseline.
        scope (str): Scope for identifying the algorithm.
            Must be specified if running multiple algorithms
            simultaneously, each using different environments
            and policies.
        max_path_length (int): Maximum length of a single rollout.
        discount (float): Discount.
        gae_lambda (float): Lambda used for generalized advantage
            estimation.
        center_adv (bool): Whether to rescale the advantages
            so that they have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift the advantages
            so that they are always positive. When used in
            conjunction with center_adv the advantages will be
            standardized before shifting.
        fixed_horizon (bool): Whether to fix horizon.
        pg_loss (str): A string from: 'vanilla', 'surrogate',
            'surrogate_clip'. The type of loss functions to use.
        lr_clip_range (float): The limit on the likelihood ratio between
            policies, as in PPO.
        max_kl_step (float): The maximum KL divergence between old and new
            policies, as in TRPO.
        optimizer (object): The optimizer of the algorithm. Should be the
            optimizers in garage.tf.optimizers.
        optimizer_args (dict): The arguments of the optimizer.
        policy_ent_coeff (float): The coefficient of the policy entropy.
            Setting it to zero would mean no entropy regularization.
        use_softplus_entropy (bool): Whether to estimate the softmax
            distribution of the entropy to prevent the entropy from being
            negative.
        use_neg_logli_entropy (bool): Whether to estimate the entropy as the
            negative log likelihood of the action.
        stop_entropy_gradient (bool): Whether to stop the entropy gradient.
        entropy_method (str): A string from: 'max', 'regularized',
            'no_entropy'. The type of entropy method to use. 'max' adds the
            dense entropy to the reward for each time step. 'regularized' adds
            the mean entropy to the surrogate objective. See
            https://arxiv.org/abs/1805.00909 for more details.
        name (str): The name of the algorithm.
    """

    def __init__(self,
                 env,
                 env_spec,
                 policy,
                 baseline,
                 expert_trajectory,
                 epochs_per_step=10,
                 max_epochs=None,
                 skip_until_step=0,
                 scope=None,
                 max_path_length=500,
                 discount=0.99,
                 gae_lambda=1,
                 center_adv=True,
                 positive_adv=False,
                 fixed_horizon=False,
                 pg_loss='surrogate_clip',
                 lr_clip_range=0.01,
                 max_kl_step=0.01,
                 optimizer=None,
                 optimizer_args=None,
                 policy_ent_coeff=0.0,
                 use_softplus_entropy=False,
                 use_neg_logli_entropy=False,
                 stop_entropy_gradient=False,
                 entropy_method='no_entropy',
                 name='PPO',
                 log_dir=None):

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

        super().__init__(
            env_spec=env_spec,
            policy=policy,
            baseline=baseline,
            scope=scope,
            max_path_length=max_path_length,
            discount=discount,
            gae_lambda=gae_lambda,
            center_adv=center_adv,
            positive_adv=positive_adv,
            fixed_horizon=fixed_horizon,
            pg_loss=pg_loss,
            lr_clip_range=lr_clip_range,
            max_kl_step=max_kl_step,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            policy_ent_coeff=policy_ent_coeff,
            use_softplus_entropy=use_softplus_entropy,
            use_neg_logli_entropy=use_neg_logli_entropy,
            stop_entropy_gradient=stop_entropy_gradient,
            entropy_method=entropy_method,
            name=name,
            # log_dir=log_dir
        )

    def train(self, runner):

        last_return = None

        max_reward = -np.inf
        max_reward_step = -1
        max_final_reward = -np.inf
        expert_trajectory_reward = np.sum(np.array([step['reward'] for step in self.expert_trajectory]))

        full_paths = []
        runner.train_args.n_epochs = self.max_epochs
        # done = False
        for epoch in self.get_next_epoch(runner=runner):
            # Get the rollouts

            epoch_itr = epoch[0]
            epoch_paths = epoch[1]
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
        paths = self.process_samples(itr, paths)

        self.log_diagnostics(paths)
        logger.log('Optimizing policy...')
        self.optimize_policy(itr, paths)
        return paths

    def get_next_epoch(self, runner):
        """ Wrapper of garage's runner.step_epochs() generator to handle initialization to correct trajectory state"""
        try:
            iteration_num = self.first_iteration_num
            self.step_num = self.first_step_num
            epochs_per_this_step = 0
            self.done = False
            self.set_env_to_expert_trajectory_step()
            self.done_with_step = False

            for epoch_num in itertools.takewhile(lambda x: not self.done, runner.step_epochs()):

                yield (runner.step_itr, runner.obtain_samples(runner.step_itr))

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
        self.env_state = self.expert_trajectory[self.step_num]['state']
        self.env_reward = np.array([step['reward'] for step in self.expert_trajectory[:self.step_num]])
        self.env_action = np.array([step['action'] for step in self.expert_trajectory[:self.step_num]])
        self.env_observation = np.array([step['observation'] for step in self.expert_trajectory[:self.step_num]])

        self.env.set_param_values([self.env_state], robustify_state=True, debug=False)
