# Import the example classes
from mylab.simulators.example_av_simulator import ExampleAVSimulator
from mylab.rewards.example_av_reward import ExampleAVReward
from mylab.spaces.example_av_spaces import ExampleAVSpaces

# Import the AST classes
from mylab.envs.ast_env import ASTEnv
from mylab.samplers.ast_vectorized_sampler import ASTVectorizedSampler

# Import the necessary garage classes
from garage.tf.algos.trpo import TRPO
from garage.tf.algos.ppo import PPO
from garage.tf.envs.base import TfEnv
from garage.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
from garage.tf.policies.uniform_control_policy import UniformControlPolicy
from garage.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from garage.np.baselines.linear_feature_baseline import LinearFeatureBaseline
from garage.envs.normalized_env import normalize
from garage.experiment import run_experiment
from garage.tf.experiment import LocalTFRunner

# Useful imports
import os.path as osp
import argparse
from example_save_trials import *
import tensorflow as tf
import fire

# parser = argparse.ArgumentParser()
# parser.add_argument('--snapshot_mode', type=str, default="gap")
# parser.add_argument('--snapshot_gap', type=int, default=10)
# parser.add_argument('--log_dir', type=str, default='./data')
# parser.add_argument('--iters', type=int, default=101)
# args = parser.parse_args()
#
# log_dir = args.log_dir

# batch_size = 5000
# max_path_length = 50
# n_envs = batch_size // max_path_length


# def runner(snapshot_mode='gap',
#            snapshot_gap=10,
#            log_dir='.',
#            n_itr=101,
#            s_0=[0.0, -4.0, 1.0, 11.17, -35.0],
#            n_parallel=1,
#            exp_name='crosswalk',
#            batch_size=None,
#
#
# ):
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
           # log_dir='.',
           ):


    if env_args is None:
        env_args = {}

    if run_experiment_args is None:
        run_experiment_args = {}

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

    if algo_args is None:
        algo_args = {}

    if runner_args is None:
        runner_args = {'n_epochs':1}

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

    def run_task(snapshot_config, *_):


        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            with tf.variable_scope('AST', reuse=tf.AUTO_REUSE):

                with LocalTFRunner(
                        snapshot_config=snapshot_config, max_cpus=4, sess=sess) as local_runner:
                    # Instantiate the example classes
                    sim = ExampleAVSimulator(**sim_args)
                    reward_function = ExampleAVReward(**reward_args)
                    spaces = ExampleAVSpaces(**spaces_args)

                    # Create the environment
                    if 'id' in env_args:
                        env_args.pop('id')
                    env = TfEnv(normalize(ASTEnv(simulator=sim,
                                                 reward_function=reward_function,
                                                 spaces=spaces,
                                                 **env_args
                                                 )))

                    # Instantiate the garage objects
                    policy = GaussianLSTMPolicy(env_spec=env.spec, **policy_args)
                                                # name='lstm_policy',
                                                # env_spec=env.spec,
                                                # hidden_dim=64,
                                                # 'use_peepholes=True)


                    baseline = LinearFeatureBaseline(env_spec=env.spec, **baseline_args)

                    optimizer = ConjugateGradientOptimizer
                    optimizer_args = {'hvp_approach': FiniteDifferenceHvp(base_eps=1e-5)}

                    algo = PPO(env_spec=env.spec,
                               policy=policy,
                               baseline=baseline,
                               optimizer=optimizer,
                               optimizer_args=optimizer_args,
                               **algo_args)
                        # max_path_length=max_path_length,
                        # discount=0.99,
                        # # kl_constraint='hard',
                        # optimizer=optimizer,
                        # optimizer_args=optimizer_args,
                        # lr_clip_range=1.0,
                        # max_kl_step=1.0)

                    sampler_cls = ASTVectorizedSampler

                    local_runner.setup(
                        algo=algo,
                        env=env,
                        sampler_cls=sampler_cls,
                        sampler_args={"open_loop": False,
                                      "sim": sim,
                                      "reward_function": reward_function,
                                      'n_envs': n_parallel})

                    # Run the experiment
                    local_runner.train(**runner_args)

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
                    # if args.snapshot_mode != "gap":
                    #     args.snapshot_gap = args.iters - 1
                    # example_save_trials(args.iters, args.log_dir, header, sess, save_every_n=args.snapshot_gap)



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