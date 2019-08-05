# Import the example classes
from mylab.simulators.example_av_simulator import ExampleAVSimulator
from mylab.rewards.example_av_reward import ExampleAVReward
from mylab.spaces.example_av_spaces import ExampleAVSpaces

# Import the AST classes
from mylab.envs.ast_env import ASTEnv
from mylab.samplers.ast_vectorized_sampler import ASTVectorizedSampler

# Import the necessary garage classes
from garage.tf.algos.trpo import TRPO
from garage.tf.envs.base import TfEnv
from garage.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
from garage.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from garage.np.baselines.linear_feature_baseline import LinearFeatureBaseline
from garage.envs.normalized_env import normalize
from garage.experiment import LocalRunner, run_experiment


# Useful imports
import os.path as osp
import argparse
from example_save_trials import *
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--snapshot_mode', type=str, default="gap")
parser.add_argument('--snapshot_gap', type=int, default=10)
parser.add_argument('--log_dir', type=str, default='../data/')
parser.add_argument('--iters', type=int, default=1)
args = parser.parse_args()

log_dir = args.log_dir

batch_size = 4000
max_path_length = 50
n_envs = batch_size // max_path_length


def run_task(snapshot_config, *_):


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        with tf.variable_scope('AST', reuse=tf.AUTO_REUSE):

            with LocalRunner(
                    snapshot_config=snapshot_config, max_cpus=n_envs, sess=sess) as runner:

                # Instantiate the example classes
                sim = ExampleAVSimulator()
                reward_function = ExampleAVReward()
                spaces = ExampleAVSpaces()

                # Create the environment
                env = TfEnv(normalize(ASTEnv(action_only=True,
                                             fixed_init_state=False,
                                             s_0=[-0.5, -4.0, 1.0, 11.17, -35.0],
                                             simulator=sim,
                                             reward_function=reward_function,
                                             spaces=spaces
                                             )))

                # Instantiate the garage objects
                policy = GaussianLSTMPolicy(name='lstm_policy',
                                            env_spec=env.spec,
                                            hidden_dim=64,
                                            use_peepholes=True)

                baseline = LinearFeatureBaseline(env_spec=env.spec)

                optimizer = ConjugateGradientOptimizer
                optimizer_args = {'hvp_approach': FiniteDifferenceHvp(base_eps=1e-5)}

                algo = TRPO(
                    env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    max_path_length=max_path_length,
                    discount=0.99,
                    kl_constraint='hard',
                    optimizer=optimizer,
                    optimizer_args=optimizer_args,
                    max_kl_step=0.01)

                sampler_cls = ASTVectorizedSampler

                runner.setup(
                    algo=algo,
                    env=env,
                    sampler_cls=sampler_cls,
                    sampler_args={"sim": sim,
                                  "reward_function": reward_function})

                # Run the experiment
                runner.train(n_epochs=args.iters, batch_size=4000, plot=False)

                saver = tf.train.Saver()
                save_path = saver.save(sess, log_dir + '/model.ckpt')
                print("Model saved in path: %s" % save_path)

                # Write out the episode results
                header = 'trial, step, ' + 'v_x_car, v_y_car, x_car, y_car, '
                for i in range(0,sim.c_num_peds):
                    header += 'v_x_ped_' + str(i) + ','
                    header += 'v_y_ped_' + str(i) + ','
                    header += 'x_ped_' + str(i) + ','
                    header += 'y_ped_' + str(i) + ','

                for i in range(0,sim.c_num_peds):
                    header += 'a_x_'  + str(i) + ','
                    header += 'a_y_' + str(i) + ','
                    header += 'noise_v_x_' + str(i) + ','
                    header += 'noise_v_y_' + str(i) + ','
                    header += 'noise_x_' + str(i) + ','
                    header += 'noise_y_' + str(i) + ','

                header += 'reward'
                if args.snapshot_mode != "gap":
                    args.snapshot_gap = args.iters - 1
                example_save_trials(args.iters, args.log_dir, header, sess, save_every_n=args.snapshot_gap)



run_experiment(run_task, snapshot_mode='last', seed=1)