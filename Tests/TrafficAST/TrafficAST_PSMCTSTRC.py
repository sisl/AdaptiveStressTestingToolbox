import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU

# from garage.tf.algos.trpo import TRPO
from garage.baselines.zero_baseline import ZeroBaseline
from mylab.envs.tfenv import TfEnv
from garage.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from garage.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
from garage.tf.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from garage.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from garage.misc import logger
from garage.envs.normalized_env import normalize
from garage.envs.env_spec import EnvSpec

from mylab.algos.psmctstrc import PSMCTSTRC

import os.path as osp
import argparse
# from example_save_trials import *
import tensorflow as tf
import joblib
import math
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default="cartpole")
parser.add_argument('--itr', type=int, default=20000)
parser.add_argument('--k',type=float, default=0.5)
parser.add_argument('--alpha',type=float, default=0.5)
parser.add_argument('--ec',type=float, default=10.0)#1.414)
parser.add_argument('--initial_pop',type=int, default=0)
parser.add_argument('--ca',type=int, default=4) # number of candicate action
parser.add_argument('--lr', type=float, default=1.0)
parser.add_argument('--lr_anneal', type=float, default=1.0)
parser.add_argument('--f_F',type=str, default="mean")
parser.add_argument('--f_Q', type=str, default='max')
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--plot_tree', type=bool, default=False)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--snapshot_mode', type=str, default="gap")
parser.add_argument('--snapshot_gap', type=int, default=5000)
parser.add_argument('--log_dir', type=str, default='./Data/AST/PSMCTSTRC')
parser.add_argument('--args_data', type=str, default=None)
args = parser.parse_args()

# Create the logger
log_dir = args.log_dir\
            +('K'+str(args.k))\
            +('A'+str(args.alpha))\
            +('Ec'+str(args.ec))\
            +('InitP'+str(args.initial_pop) if (args.initial_pop>0) else '')\
            +('CA'+str(args.ca))\
            +('lr'+str(args.lr))\
            +('anneal'+str(args.lr_anneal) if (args.lr_anneal != 1.) else '')\
            +('F'+args.f_F)\
            +('Q'+args.f_Q)\
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
max_path_length = 100

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
    policy = DeterministicMLPPolicy(
        name='ast_agent',
        env_spec=env.spec,
        hidden_sizes=(64, 32)
    )

    params = policy.get_params()
    sess.run(tf.variables_initializer(params))

    # Instantiate the RLLAB objects
    baseline = ZeroBaseline(env_spec=env.spec) # not used
    # optimizer = ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))

    algo = PSMCTSTRC(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=max_path_length,
        step_size=args.lr,
        step_size_anneal=args.lr_anneal,
        n_itr=args.itr+1,
        max_path_length=max_path_length,
        top_paths=top_paths,
        ec=args.ec,
        k=args.k,
        alpha=args.alpha,
        log_interval=args.log_interval,
        plot=False,
        n_ca=args.ca,
        initial_pop=args.initial_pop,
        f_F=args.f_F,
        f_Q=args.f_Q,
        )

    algo.train(sess=sess, init_var=False)
    if args.plot_tree:
        from mylab.utils.tree_plot import plot_tree
        plot_tree(algo.s,d=max_path_length,path=log_dir+"/tree",format="png")

    