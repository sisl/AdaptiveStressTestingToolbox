# Import the example classes
import compress_pickle
import fire
import gym
# Useful imports
import tensorflow as tf
from garage.envs.normalized_env import normalize
from garage.experiment import run_experiment
from garage.np.baselines.linear_feature_baseline import LinearFeatureBaseline
from garage.tf.envs.base import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from garage.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from garage.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy

# Import the necessary garage classes
from ast_toolbox.algos import BackwardAlgorithm
from ast_toolbox.rewards import ExampleAVReward
from ast_toolbox.samplers import BatchSampler
from ast_toolbox.simulators import ExampleAVSimulator
from ast_toolbox.spaces import ExampleAVSpaces

# Import the AST classes


#
# parser = argparse.ArgumentParser()
# parser.add_argument('--snapshot_mode', type=str, default="gap")
# parser.add_argument('--snapshot_gap', type=int, default=10)
# parser.add_argument('--log_dir', type=str, default='../data/')
# parser.add_argument('--iters', type=int, default=1)
# args = parser.parse_args()
#
# log_dir = args.log_dir
#
# batch_size = 4000
# max_path_length = 50
# n_envs = batch_size // max_path_length

# Example run command:
# python examples/AV/example_runner_ge_av.py runner --n_itr=1

def runner(
        env_args=None,
        run_experiment_args=None,
        sim_args=None,
        reward_args=None,
        spaces_args=None,
        policy_args=None,
        baseline_args=None,
        algo_args=None,
        runner_args=None,
        sampler_args=None,
        # log_dir='.',
):
    if not isinstance(env_args, dict) or 'id' not in env_args.keys():
        print('ERROR: Must supply an environment id in env_args')
        raise Exception

    if run_experiment_args is None:
        run_experiment_args = {'log_dir': '.'}

    if sim_args is None:
        sim_args = {}

    if reward_args is None:
        reward_args = {}

    if spaces_args is None:
        spaces_args = {}

    if policy_args is None:
        policy_args = {}

    if baseline_args is None:
        baseline_args = {}

    if algo_args is dict and 'id' not in env_args.keys():
        print('ERROR: Must supply an expert trajectory')
        raise Exception

    if runner_args is None:
        runner_args = {'n_epochs': 1}

    if sampler_args is None:
        sampler_args = {}

    if 'n_parallel' in run_experiment_args:
        n_parallel = run_experiment_args['n_parallel']
    else:
        n_parallel = 1
        run_experiment_args['n_parallel'] = n_parallel

    if 'max_path_length' in sim_args:
        max_path_length = sim_args['max_path_length']
    else:
        max_path_length = 50
        sim_args['max_path_length'] = max_path_length

    if 'batch_size' in runner_args:
        batch_size = runner_args['batch_size']
    else:
        batch_size = max_path_length * n_parallel
        runner_args['batch_size'] = batch_size

    # def runner(exp_name='av',
    #            # use_ram=False,
    #            # db_filename='/home/mkoren/scratch/data/cellpool-shelf',
    #            # max_db_size=150,
    #            # overwrite_db=True,
    #            # expert_trajectory_file='/home/mkoren/scratch/data/test1_sparse_ge/expert_trajectory.p',
    #            n_parallel=1,
    #            snapshot_mode='last',
    #            snapshot_gap=1,
    #            log_dir=None,
    #            max_path_length=50,
    #            discount=0.99,
    #            n_itr=100,
    #            max_kl_step=0.01,
    #            whole_paths=False,
    #            batch_size=None,):
    #
    # if overwrite_db:
    #     with contextlib.suppress(FileNotFoundError):
    #         os.remove(db_filename +'_pool.dat')
    #         os.remove(db_filename + '_meta.dat')

    def run_task(snapshot_config, *_):

        config = tf.ConfigProto(device_count={'GPU': 0})
        # config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            with tf.variable_scope('AST', reuse=tf.AUTO_REUSE):
                with LocalTFRunner(
                        snapshot_config=snapshot_config, sess=sess) as local_runner:
                    # Instantiate the example classes
                    sim = ExampleAVSimulator(**sim_args)
                    reward_function = ExampleAVReward(**reward_args)
                    spaces = ExampleAVSpaces(**spaces_args)

                    # Create the environment
                    # env1 = GoExploreASTEnv(open_loop=False,
                    #                              blackbox_sim_state=True,
                    #                              fixed_init_state=True,
                    #                              s_0=[-0.5, -4.0, 1.0, 11.17, -35.0],
                    #                              simulator=sim,
                    #                              reward_function=reward_function,
                    #                              spaces=spaces
                    #                              )
                    env1 = gym.make(id=env_args.pop('id'),
                                    simulator=sim,
                                    reward_function=reward_function,
                                    spaces=spaces,
                                    **env_args)
                    env2 = normalize(env1)
                    env = TfEnv(env2)

                    sampler_cls = BatchSampler
                    # sampler_args = {'n_envs': n_parallel}
                    # sampler_args = {"open_loop": env_args['open_loop'],
                    #                 "sim": sim,
                    #                 "reward_function": reward_function,
                    #                 'n_envs': n_parallel}
                    sampler_args['sim'] = sim
                    sampler_args['reward_function'] = reward_function

                    # expert_trajectory_file = log_dir + '/expert_trajectory.p'
                    # with open(expert_trajectory_file, 'rb') as f:
                    #     expert_trajectory = pickle.load(f)

                    #
                    # #Run backwards algorithm to robustify
                # with LocalTFRunner(
                #         snapshot_config=snapshot_config, sess=sess) as local_runner:

                    policy = GaussianLSTMPolicy(env_spec=env.spec, **policy_args)
                    # name='lstm_policy',
                    # env_spec=env.spec,
                    # hidden_dim=64,
                    # use_peepholes=True)

                    baseline = LinearFeatureBaseline(env_spec=env.spec, **baseline_args)

                    optimizer = ConjugateGradientOptimizer
                    optimizer_args = {'hvp_approach': FiniteDifferenceHvp(base_eps=1e-5)}

                    algo = BackwardAlgorithm(env=env,
                                             env_spec=env.spec,
                                             policy=policy,
                                             baseline=baseline,
                                             optimizer=optimizer,
                                             optimizer_args=optimizer_args,
                                             **algo_args)
                    # expert_trajectory=expert_trajectory[-1],
                    # epochs_per_step = 10,
                    # scope=None,
                    # max_path_length=max_path_length,
                    # discount=discount,
                    # gae_lambda=1,
                    # center_adv=True,
                    # positive_adv=False,
                    # fixed_horizon=False,
                    # pg_loss='surrogate_clip',
                    # lr_clip_range=1.0,
                    # max_kl_step=1.0,

                    # policy_ent_coeff=0.0,
                    # use_softplus_entropy=False,
                    # use_neg_logli_entropy=False,
                    # stop_entropy_gradient=False,
                    # entropy_method='no_entropy',
                    # name='PPO',
                    # )

                    local_runner.setup(algo=algo,
                                       env=env,
                                       sampler_cls=sampler_cls,
                                       sampler_args=sampler_args)

                    results = local_runner.train(**runner_args)
                    # pdb.set_trace()
                    print('done')
                    log_dir = run_experiment_args['log_dir']
                    with open(log_dir + '/paths.gz', 'wb') as f:
                        try:
                            compress_pickle.dump(results, f, compression="gzip", set_default_extension=False)
                        except MemoryError:
                            print('1')
                            # pdb.set_trace()
                            for idx, result in enumerate(results):
                                with open(log_dir + '/path_' + str(idx) + '.gz', 'wb') as ff:
                                    try:
                                        compress_pickle.dump(result, ff, compression="gzip",
                                                             set_default_extension=False)
                                    except MemoryError:
                                        print('2')

                    # saver = tf.train.Saver()
                    # save_path = saver.save(sess, log_dir + '/model.ckpt')
                    # print("Model saved in path: %s" % save_path)
                    #
                    # # Write out the episode results
                    # header = 'trial, step, ' + 'v_x_car, v_y_car, x_car, y_car, '
                    # for i in range(0,sim.c_num_peds):
                    #     header += 'v_x_ped_' + str(i) + ','
                    #     header += 'v_y_ped_' + str(i) + ','
                    #     header += 'x_ped_' + str(i) + ','
                    #     header += 'y_ped_' + str(i) + ','
                    #
                    # for i in range(0,sim.c_num_peds):
                    #     header += 'a_x_'  + str(i) + ','
                    #     header += 'a_y_' + str(i) + ','
                    #     header += 'noise_v_x_' + str(i) + ','
                    #     header += 'noise_v_y_' + str(i) + ','
                    #     header += 'noise_x_' + str(i) + ','
                    #     header += 'noise_y_' + str(i) + ','
                    #
                    # header += 'reward'
                    # if snapshot_mode != "gap":
                    #     snapshot_gap = n_itr - 1
                    # example_save_trials(n_itr, log_dir, header, sess, save_every_n=snapshot_gap)

    # run_experiment(
    #     run_task,
    #     snapshot_mode=snapshot_mode,
    #     log_dir=log_dir,
    #     exp_name=exp_name,
    #     snapshot_gap=snapshot_gap,
    #     seed=1,
    #     n_parallel=n_parallel,
    #     tabular_log_file='progress_ba.csv'
    # )
    run_experiment(
        run_task,
        **run_experiment_args,
        # snapshot_mode=snapshot_mode,
        # log_dir=log_dir,
        # exp_name=exp_name,
        # snapshot_gap=snapshot_gap,
        # seed=1,
        # n_parallel=n_parallel,
    )


if __name__ == '__main__':
    fire.Fire()
