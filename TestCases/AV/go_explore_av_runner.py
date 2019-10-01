# Import the example classes
from mylab.simulators.example_av_simulator import ExampleAVSimulator
from mylab.rewards.example_av_reward import ExampleAVReward
from mylab.spaces.example_av_spaces import ExampleAVSpaces

# Import the AST classes
from mylab.envs.go_explore_ast_env import GoExploreASTEnv, Custom_GoExploreASTEnv
from mylab.samplers.ast_vectorized_sampler import ASTVectorizedSampler

# Import the necessary garage classes
from mylab.algos.go_explore import GoExplore
from garage.tf.envs.base import TfEnv
from mylab.policies.go_explore_policy import GoExplorePolicy
from garage.np.baselines.linear_feature_baseline import LinearFeatureBaseline
from garage.envs.normalized_env import normalize
from garage.experiment import LocalRunner, run_experiment
from mylab.samplers.batch_sampler import BatchSampler
import gym

# Useful imports
from example_save_trials import *
import tensorflow as tf
import fire
import os
import pdb

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

                with LocalRunner(
                        snapshot_config=snapshot_config, sess=sess) as runner:

                    # Instantiate the example classes
                    sim = ExampleAVSimulator()
                    reward_function = ExampleAVReward()
                    spaces = ExampleAVSpaces()

                    # Create the environment
                    # env1 = GoExploreASTEnv(open_loop=False,
                    #                              action_only=True,
                    #                              fixed_init_state=True,
                    #                              s_0=[-0.5, -4.0, 1.0, 11.17, -35.0],
                    #                              simulator=sim,
                    #                              reward_function=reward_function,
                    #                              spaces=spaces
                    #                              )
                    env1 = gym.make('mylab:GoExploreAST-v1',
                             open_loop=False,
                             action_only=True,
                             fixed_init_state=True,
                             s_0=[-0.5, -4.0, 1.0, 11.17, -35.0],
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