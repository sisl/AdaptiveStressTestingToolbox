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
from garage.baselines.linear_feature_baseline import LinearFeatureBaseline
from garage.envs.normalized_env import normalize
from garage.misc import logger

# Useful imports
import os.path as osp
import argparse
from example_save_trials import *
import tensorflow as tf

# Logger Params
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='crosswalk_exp')
parser.add_argument('--tabular_log_file', type=str, default='tab.txt')
parser.add_argument('--text_log_file', type=str, default='tex.txt')
parser.add_argument('--params_log_file', type=str, default='args.txt')
parser.add_argument('--snapshot_mode', type=str, default="gap")
parser.add_argument('--snapshot_gap', type=int, default=10)
parser.add_argument('--log_tabular_only', type=bool, default=False)
parser.add_argument('--log_dir', type=str, default='.')
parser.add_argument('--args_data', type=str, default=None)

#Algo Params
parser.add_argument('--iters', type=int, default=101)
parser.add_argument('--batch_size', type=int, default=4000)
parser.add_argument('--step_size', type=float, default=1.0)


# Policy Params
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--policy', type=str, default="LSTM")
parser.add_argument('--use_peepholes', type=bool, default=False)
parser.add_argument('--load_policy', type=bool, default=False)

# Env Params
parser.add_argument('--action_only', type=bool, default=True)
parser.add_argument('--sample_init_state', type=bool, default=False)

# Parse input args
args = parser.parse_args()

# Create the logger
log_dir = args.log_dir

tabular_log_file = osp.join(log_dir, args.tabular_log_file)
text_log_file = osp.join(log_dir, args.text_log_file)
params_log_file = osp.join(log_dir, args.params_log_file)

# logger = Logger()

logger.log_parameters_lite(params_log_file, args)
logger.add_text_output(text_log_file)
logger.add_tabular_output(tabular_log_file)
prev_snapshot_dir = logger.get_snapshot_dir()
prev_mode = logger.get_snapshot_mode()
logger.set_snapshot_dir(log_dir)
logger.set_snapshot_mode(args.snapshot_mode)
logger.set_snapshot_gap(args.snapshot_gap)
logger.set_log_tabular_only(args.log_tabular_only)
logger.push_prefix("[%s] " % args.exp_name)

# Instantiate the example classes
sim = ExampleAVSimulator()
reward_function = ExampleAVReward()
spaces = ExampleAVSpaces()

# Create the environment
env = TfEnv(normalize(ASTEnv(action_only=args.action_only,
                             sample_init_state=args.sample_init_state,
                             s_0=[0.0, -2.0, 1.0, 11.17, -35.0],
                             simulator=sim,
                             reward_function=reward_function,
                             spaces=spaces
                             )))
print("Number of policy parameters: ",
      4*(args.hidden_dim**2 + args.hidden_dim*(
          spaces.action_space.flat_dim +
          spaces.observation_space.flat_dim) +
         args.hidden_dim))
# Instantiate the garage objects
policy = GaussianLSTMPolicy(name='lstm_policy',
                            env_spec=env.spec,
                            hidden_dim=args.hidden_dim,
                            use_peepholes=args.use_peepholes)

baseline = LinearFeatureBaseline(env_spec=env.spec)
optimizer = ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
sampler_cls = ASTVectorizedSampler
algo = TRPO(
    env=env,
    policy=policy,
    baseline=LinearFeatureBaseline(env_spec=env.spec),
    batch_size=args.batch_size,
    step_size=args.step_size,
    n_itr=args.iters,
    store_paths=True,
    optimizer=optimizer,
    max_path_length=50,
    sampler_cls=sampler_cls,
    sampler_args={"sim": sim,
                  "reward_function": reward_function})

saver = tf.train.Saver()

with tf.Session() as sess:
    # Run the experiment
    algo.train(sess=sess)
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
    example_save_trials(algo.n_itr, args.log_dir, header, sess, save_every_n=args.snapshot_gap)


