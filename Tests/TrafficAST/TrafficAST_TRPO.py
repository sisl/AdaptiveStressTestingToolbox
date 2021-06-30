import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU

# from garage.tf.algos.trpo import TRPO
from garage.baselines.zero_baseline import ZeroBaseline
from mylab.envs.tfenv import TfEnv
from garage.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from garage.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
from garage.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from garage.misc import logger
from garage.envs.normalized_env import normalize
from garage.envs.env_spec import EnvSpec

import os.path as osp
import argparse
# from example_save_trials import *
import tensorflow as tf
import joblib
import math
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default="TrafficAST-TRPO")
parser.add_argument('--inita', type=float, default=0.5)
parser.add_argument('--itr', type=int, default=1000)
parser.add_argument('--bs', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--snapshot_mode', type=str, default="gap")
parser.add_argument('--snapshot_gap', type=int, default=500)
parser.add_argument('--log_dir', type=str, default='TRPO')
parser.add_argument('--args_data', type=str, default=None)
args = parser.parse_args()

# Create the logger
pre_dir = './Data/AST_inita{}/'.format(args.inita)
log_dir = pre_dir+args.log_dir\
            +('bs'+str(args.bs))\
            +('lr'+str(args.lr))\
            +('/'+'seed'+str(args.seed))
args.log_dir = log_dir

tabular_log_file = osp.join(log_dir, 'progress.csv')
text_log_file = osp.join(log_dir, 'text.txt')
params_log_file = osp.join(log_dir, 'args.txt')

logger.log_parameters_lite(params_log_file, args)
# logger.add_text_output(text_log_file)
logger.add_tabular_output(tabular_log_file)
logger.set_snapshot_dir(log_dir)
logger.set_snapshot_mode(args.snapshot_mode)
logger.set_snapshot_gap(args.snapshot_gap)
logger.set_log_tabular_only(False)
logger.push_prefix("[%s] " % args.exp_name)

seed = args.seed
top_k = 10

import mcts.BoundedPriorityQueues as BPQ
top_paths = BPQ.BoundedPriorityQueue(top_k)

np.random.seed(seed)
tf.set_random_seed(seed)
with tf.Session() as sess:
    # Create env
    from traffic.make_env import make_env
    env_inner = make_env(env_name='highway')
    data = joblib.load("Data/Train/TRPO/seed0/itr_1000.pkl")
    policy_inner = data['policy']

    from mylab.rewards.ast_reward import ASTReward
    from mylab.envs.ast_env import ASTEnv
    from mylab.simulators.policy_simulator import PolicySimulator
    reward_function = ASTReward()
    simulator = PolicySimulator(env=env_inner,policy=policy_inner,max_path_length=100)
    env = TfEnv(ASTEnv(interactive=True,
                                 simulator=simulator,
                                 sample_init_state=False,
                                 s_0=0., # not used
                                 reward_function=reward_function,
                                 ))

    # Create policy
    policy = GaussianMLPPolicy(
        name='Policy',
        env_spec=env.spec,
        hidden_sizes=(64, 32)
    )
    
    params = policy.get_params()
    sess.run(tf.variables_initializer(params))

    # Instantiate the RLLAB objects
    # from garage.baselines.zero_baseline import ZeroBaseline
    # baseline = ZeroBaseline(env_spec=env.spec)
    from garage.baselines.linear_feature_baseline import LinearFeatureBaseline
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    # from garage.tf.baselines.deterministic_mlp_baseline import DeterministicMLPBaseline
    # baseline = DeterministicMLPBaseline(env_spec=env.spec,
    #                                 regressor_args={
    #                                     'hidden_sizes':(32, 32),
    #                                     'hidden_nonlinearity':tf.nn.tanh,
    #                                 }) # unstable

    from mylab.algos.trpo import TRPO
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=args.bs,
        step_size=args.lr,
        n_itr=args.itr+1,
        store_paths=False,
        max_path_length=100,
        top_paths = top_paths,
        plot=False,
        )

    algo.train(sess=sess, init_var=False)
    