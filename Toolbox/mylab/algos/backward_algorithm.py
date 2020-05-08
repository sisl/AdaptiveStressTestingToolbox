from garage.tf.algos.ppo import PPO
from garage.tf.algos.trpo import TRPO
from garage.tf.optimizers import FirstOrderOptimizer
from dowel import logger, tabular
import numpy as np
import pdb

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
                 epochs_per_step = 10,
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
        # if optimizer is None:
        #     optimizer = FirstOrderOptimizer
        #     if optimizer_args is None:
        #         optimizer_args = dict()

        self.max_epochs_per_step = epochs_per_step
        # self.max_steps = len(expert_trajectory)
        # self.skip_until_step = int(.5 * self.max_steps)
        self.max_steps = max_path_length
        self.skip_until_step = 5
        self.expert_trajectory = expert_trajectory

        self.env = env
        self.policy = policy

        self.env.set_param_values([None], robustify_state=True, debug=False)

        # self.cell_pool = CellPool(filename=self.db_filename, flag=db.DB_CREATE, flag2='n')
        #
        #
        # pool_DB = db.DB()
        # pool_DB.open(self.db_filename, dbname=None, dbtype=db.DB_HASH, flags=db.DB_CREATE)
        # d_pool = shelve.Shelf(pool_DB, protocol=pickle.HIGHEST_PROTOCOL)
        # obs, state = self.env.get_first_cell()
        #
        # self.cell_pool.d_update(d_pool=d_pool, observation=obs, trajectory=np.array([]), score=0.0, state=state, reward=0.0, chosen=1)
        # d_pool.sync()
        #
        #
        # self.env.set_param_values([self.db_filename], db_filename=True, debug=False)
        # self.env.set_param_values([self.cell_pool.key_list], key_list=True, debug=False)
        # self.env.set_param_values([self.cell_pool.max_value], max_value=True, debug=False)
        # self.env.set_param_values([None], robustify_state=True, debug=False)
        # d_pool.close()

        # pdb.set_trace()

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

    # def train(self, runner, batch_size):
    #     # pdb.set_trace()
    #     last_return = None
    #     for _ in runner.step_epochs():
    #         #Just need to instantiate the step_epochs generator
    #         for step_num, env_state in enumerate(self.expert_trajectory):
    #             if step_num <= self.skip_until_step:
    #                 continue
    #             # pdb.set_trace()
    #             #Set environment to reset to the correct step:
    #             self.env.set_param_values([env_state], robustify_state=True, debug=False)
    #             # self.max_path_length = step_num
    #             for epoch in range(self.epochs_per_step):
    #                 # pdb.set_trace()
    #                 runner.step_path = runner.obtain_samples(runner.step_itr, batch_size)
    #                 last_return = self.train_once(runner.step_itr, runner.step_path)
    #                 runner.step_itr += 1
    #
    #         return last_return
    #     return last_return

    def train(self, runner):
        # pdb.set_trace()
        last_return = None

        # set epoch, step numbers and maximums
        step_num = np.minimum(self.skip_until_step, len(self.expert_trajectory)-1)
        num_steps = len(self.expert_trajectory) - step_num
        max_epochs = runner.train_args.n_epochs
        self.max_epochs_per_step = max_epochs // num_steps
        # Init environment to run for first step of expert trajectory
        epochs_per_this_step = 0
        env_state = self.expert_trajectory[step_num]['state']
        env_reward = self.expert_trajectory[step_num]['reward']
        env_action = self.expert_trajectory[step_num]['action']
        env_observation = self.expert_trajectory[step_num]['observation']
        self.env.set_param_values([env_state], robustify_state=True, debug=False)

        initial_reward = env_reward
        max_reward = -np.inf
        max_reward_step = -1
        max_final_reward = -np.inf

        full_paths = []
        for epoch in runner.step_epochs():

            # print('Policy stddev: ', policy)
            if epochs_per_this_step == self.max_epochs_per_step:
                # Back up the algorithm to the next step of the expert trajectory
                epochs_per_this_step = 0
                # print('------------ Backward Algorithm: Stepping Back ------------------')
                step_num = np.minimum(step_num + 1, num_steps - 1)
                print(step_num)
                env_state = self.expert_trajectory[step_num]['state']
                env_reward = self.expert_trajectory[step_num]['reward']
                env_action = self.expert_trajectory[step_num]['action']
                env_observation = self.expert_trajectory[step_num]['observation']
                self.env.set_param_values([env_state], robustify_state=True, debug=False)
                print(self.env.get_param_values())

            runner.step_path = runner.obtain_samples(runner.step_itr)

            # Set first step to that of the base step
            for rollout_idx, rollout in enumerate(runner.step_path):
                # pdb.set_trace()
                # if rollout['rewards'].shape[0] == self.max_path_length:
                    # runner.step_path[rollout_idx]['rewards'][0] = env_reward + rollout['rewards'][0]
                # else:
                if rollout['rewards'].shape[0] < self.max_path_length:
                    runner.step_path[rollout_idx]['rewards'] = np.concatenate(([env_reward], rollout['rewards']))
                    runner.step_path[rollout_idx]['actions'] = np.concatenate((np.expand_dims(env_action,axis=0),rollout['actions']))
                    runner.step_path[rollout_idx]['observations'] = np.concatenate((np.expand_dims(env_observation,axis=0),rollout['observations']))
            last_return = self.train_once(runner.step_itr, runner.step_path)
            full_paths.append(last_return)
            runner.step_itr += 1

            for path in last_return['paths']:
                if np.sum(path['rewards']) >= max_reward:
                    max_reward = np.sum(path['rewards'])
                    max_reward_step = step_num
                if step_num == num_steps - 1 and np.sum(path['rewards']) > max_final_reward:
                    max_final_reward = np.sum(path['rewards'])
            # print(np.exp(last_return['agent_infos']['log_std']))

            epochs_per_this_step += 1

        print('Backward Results -- Initial Reward: ', initial_reward, ' -- Best Reward at step ', max_reward_step, ': ', max_reward, " -- Best Final Reward: ", max_final_reward)
        return full_paths


    def train_once(self, itr, paths):
        paths = self.process_samples(itr, paths)

        self.log_diagnostics(paths)
        logger.log('Optimizing policy...')
        self.optimize_policy(itr, paths)
        return paths


        # last_return = None
        #
        # for epoch in runner.step_epochs():
        #     if runner.step_itr <= self.skip_until_step:
        #     runner.step_path = runner.obtain_samples(runner.step_itr,
        #                                              batch_size)
        #     last_return = self.train_once(runner.step_itr, runner.step_path)
        #     runner.step_itr += 1
        #
        # return last_return

    # def train_once(self, itr, paths):
    #     paths = self.process_samples(itr, paths)
    #
    #     self.log_diagnostics(paths)
    #     logger.log('Optimizing policy...')
    #     self.optimize_policy(itr, paths)
    #     return paths['average_return']