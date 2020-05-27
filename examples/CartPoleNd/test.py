import argparse
import math
import os
import os.path as osp

import joblib
import numpy as np
import src.ast_toolbox.mcts.BoundedPriorityQueues as BPQ
# from example_save_trials import *
import tensorflow as tf
from CartPole.cartpole_simulator import CartpoleSimulator
# from garage.tf.algos.trpo import TRPO
from garage.baselines.linear_feature_baseline import LinearFeatureBaseline
from garage.misc import logger
from garage.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from src.ast_toolbox import TRPO
from src.ast_toolbox import ASTEnv
from src.ast_toolbox import TfEnv
from src.ast_toolbox.rewards import ASTRewardS

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # just use CPU


# Logger Params
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='cartpole_exp')
parser.add_argument('--snapshot_mode', type=str, default="none")
parser.add_argument('--snapshot_gap', type=int, default=10)
parser.add_argument('--log_dir', type=str, default='./Data/AST/TRPO/Test')
parser.add_argument('--args_data', type=str, default=None)
args = parser.parse_args()

# Create the logger
log_dir = args.log_dir

tabular_log_file = osp.join(log_dir, 'process.csv')
text_log_file = osp.join(log_dir, 'text.csv')
params_log_file = osp.join(log_dir, 'args.txt')

logger.log_parameters_lite(params_log_file, args)
logger.add_text_output(text_log_file)
logger.add_tabular_output(tabular_log_file)
prev_snapshot_dir = logger.get_snapshot_dir()
prev_mode = logger.get_snapshot_mode()
logger.set_snapshot_dir(log_dir)
logger.set_snapshot_mode(args.snapshot_mode)
logger.set_snapshot_gap(args.snapshot_gap)
logger.set_log_tabular_only(False)
logger.push_prefix("[%s] " % args.exp_name)

seed = 0
top_k = 10

top_paths = BPQ.BoundedPriorityQueue(top_k)

np.random.seed(seed)
tf.set_random_seed(seed)
with tf.Session() as sess:
    # Create env
    # data = joblib.load("../CartPole/ControlPolicy/itr_"+str(args.sut_itr)+".pkl")
    data = joblib.load("../CartPole/Data/Train/itr_5.pkl")
    sut = data['policy']
    reward_function = ASTRewardS()
    # sut_param = np.copy(sut.get_param_values(trainable=True))

    simulator = CartpoleSimulator(sut=sut, max_path_length=100, use_seed=False, nd=1)
    env = ASTEnv(open_loop=False,
                 simulator=simulator,
                 fixed_init_state=True,
                 s_0=[0.0, 0.0, 0.0 * math.pi / 180, 0.0],
                 reward_function=reward_function,
                 )
    env = TfEnv(env)
    print(env.vectorized)
    # Create policy
    policy = GaussianMLPPolicy(
        name='ast_agent',
        env_spec=env.spec,
        hidden_sizes=(64, 32)
    )
    # policy = GaussianLSTMPolicy(name='lstm_policy',
    #                             env_spec=ast_spec,
    #                             hidden_dim=128,
    #                             use_peepholes=True)

    params = policy.get_params()
    sess.run(tf.variables_initializer(params))

    # Instantiate the garage objects
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    # optimizer = ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        step_size=0.1,
        n_itr=5,
        store_paths=True,
        # optimizer= optimizer,
        max_path_length=100,
        top_paths=top_paths,
        plot=False,
    )

    algo.train(sess=sess, init_var=False)

    # print(np.array_equal(sut_param,sut.get_param_values(trainable=True)))

    from src.ast_toolbox.algos.mcts import MCTS
    top_paths2 = BPQ.BoundedPriorityQueue(top_k)
    algo = MCTS(
        env=env,
        stress_test_num=1,
        max_path_length=100,
        ec=100.0,
        n_itr=50,
        k=0.5,
        alpha=0.5,  # 0.85,
        clear_nodes=False,
        log_interval=1000,
        top_paths=top_paths2,
        plot_tree=True,
        plot_path=args.log_dir + '/tree'
    )

    algo.train()

    import src.ast_toolbox.mcts.AdaptiveStressTesting as AST
    import src.ast_toolbox.mcts.ASTSim as ASTSim

    # ast_params = AST.ASTParams(100,0,False)
    # ast = AST.AdaptiveStressTest(p=ast_params, env=env, top_paths=top_paths)
    print("~~~~~~~~~~~~~~~~~~check TRPO reward consistance~~~~~~~~~~~~~~~")

    for (r, actions) in top_paths:
        print(np.mean(np.clip(actions, -1.0, 1.0)))
        action_seq = [AST.ASTAction(a) for a in actions]
        reward, _ = ASTSim.play_sequence(algo.ast, action_seq, sleeptime=0.0)
        print("predic reward: ", r)
        print("actual reward: ", reward)
