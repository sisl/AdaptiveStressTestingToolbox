# Import the example classes
import os
import pdb

import fire
import gym
import numpy as np
import tensorflow as tf
# Useful imports
from garage.envs.normalized_env import normalize
from garage.experiment import run_experiment
from garage.np.baselines.linear_feature_baseline import LinearFeatureBaseline
from garage.tf.envs.base import TfEnv
from garage.tf.experiment import LocalTFRunner

import ast_toolbox.simulators.rss_metrics as rss
# Import the necessary garage classes
from ast_toolbox.algos import GoExplore
from ast_toolbox.policies import GoExplorePolicy
from ast_toolbox.rewards import HeuristicReward
from ast_toolbox.rewards import PedestrianNoiseGaussian
from ast_toolbox.samplers import BatchSampler
from ast_toolbox.simulators.av_rss_simulator import AVRSSSimulator
from ast_toolbox.spaces import ExampleAVSpaces

# Import the AST classes


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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


def runner(exp_name='av',
           use_ram=False,
           db_filename='/home/mkoren/Scratch/cellpool-shelf.dat',
           max_db_size=150,
           overwrite_db=True,
           n_parallel=2,
           snapshot_mode='last',
           snapshot_gap=1,
           log_dir=None,
           max_path_length=100,
           discount=0.99,
           n_itr=100,
           max_kl_step=0.01,
           whole_paths=False,
           batch_size=None):

    if overwrite_db and os.path.exists(db_filename):
        os.remove(db_filename)

    if batch_size is None:
        batch_size = max_path_length * n_parallel

    def run_task(snapshot_config, *_):

        config = tf.ConfigProto(device_count={'GPU': 0})
        # config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            with tf.variable_scope('AST', reuse=tf.AUTO_REUSE):

                with LocalTFRunner(
                        snapshot_config=snapshot_config, sess=sess) as runner:

                    # Instantiate the example classes
                    # sim = ExampleAVSimulator()
                    g = 9.8  # acceleration due to gravity

                    # this is y
                    lat_params = rss.LateralParams(0,  # ρ
                                                   0.1 * g,  # a_lat_max_acc
                                                   0.05 * g,  # a_lat_min_brake
                                                   1.4  # Buffer distance
                                                   )

                    # this is x
                    long_params = rss.LongitudinalParams(0,  # ρ
                                                         0.7 * g,  # a_max_brake
                                                         0.1 * g,  # a_max_acc
                                                         0.7 * g,  # a_min_brake1
                                                         0.7 * g,  # a_min_brake2
                                                         2.5,  # Buffer
                                                         )
                    sim = AVRSSSimulator(lat_params, long_params)
                    reward_function = HeuristicReward(PedestrianNoiseGaussian(1, 1, 0.2, .01),
                                                      np.array([-10000, -1000, 0]))
                    # reward_function = ExampleAVReward()
                    spaces = ExampleAVSpaces()

                    # Create the environment
                    # env1 = GoExploreASTEnv(open_loop=False,
                    #                              blackbox_sim_state=True,
                    #                              fixed_init_state=True,
                    #                              s_0=[-0.5, -4.0, 1.0, 11.17, -35.0],
                    #                              simulator=sim,
                    #                              reward_function=reward_function,
                    #                              spaces=spaces
                    s_0 = [-1.0, -2.0, 1.0, 11.17, -35.0]
                    #                              )
                    env1 = gym.make('ast_toolbox:GoExploreAST-v1',
                                    open_loop=False,
                                    action_only=True,
                                    fixed_init_state=True,
                                    s_0=s_0,
                                    simulator=sim,
                                    reward_function=reward_function,
                                    spaces=spaces
                                    )
                    env2 = normalize(env1)
                    env = TfEnv(env2)

                    # Instantiate the garage objects
                    policy = GoExplorePolicy(
                        env_spec=env.spec)

                    baseline = LinearFeatureBaseline(env_spec=env.spec)

                    algo = GoExplore(
                        db_filename=db_filename,
                        max_db_size=max_db_size,
                        env=env,
                        env_spec=env.spec,
                        policy=policy,
                        baseline=baseline,
                        max_path_length=max_path_length,
                        discount=discount,
                        # whole_paths=whole_paths
                    )

                    sampler_cls = BatchSampler
                    sampler_args = {'n_envs': n_parallel}

                    runner.setup(algo=algo,
                                 env=env,
                                 sampler_cls=sampler_cls,
                                 sampler_args=sampler_args)

                    # runner.setup(
                    #     algo=algo,
                    #     env=env,
                    #     sampler_cls=sampler_cls,
                    #     sampler_args={"sim": sim,
                    #                   "reward_function": reward_function})

                    # Run the experiment
                    paths = runner.train(n_epochs=n_itr, batch_size=batch_size, plot=False)
                    print(paths)
                    best_traj = paths.trajectory * np.array([1, 1 / 1000, 1 / 1000, 1 / 1000, 1 / 1000, 1 / 1000, 1 / 1000])
                    peds = sim._peds
                    car = np.expand_dims(sim._car, axis=0)
                    car_obs = sim._car_obs
                    for step in range(best_traj.shape[0]):
                        sim.step(action=best_traj[step, 1:], open_loop=False)
                        peds = np.concatenate((peds, sim._peds), axis=0)
                        car = np.concatenate((car, np.expand_dims(sim._car, axis=0)), axis=0)
                        car_obs = np.concatenate((car_obs, sim._car_obs), axis=0)

                    import matplotlib.pyplot as plt
                    plt.scatter(car[:, 2], car[:, 3])
                    plt.scatter(peds[:, 2], peds[:, 3])
                    plt.scatter(car_obs[:, 2], car_obs[:, 3])
                    pdb.set_trace()
                    print('done!')
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

    run_experiment(
        run_task,
        snapshot_mode=snapshot_mode,
        log_dir=log_dir,
        exp_name=exp_name,
        snapshot_gap=snapshot_gap,
        seed=1,
        n_parallel=n_parallel,
    )


if __name__ == '__main__':
    fire.Fire()
