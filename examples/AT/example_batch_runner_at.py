# NOTE, run this from the root AdaptiveStressTestingToolbox directory.

import pickle

from examples.AT.example_runner_drl_at import runner as drl_runner
from examples.AT.example_runner_mcts_at import runner as mcts_runner
from examples.AT.example_runner_random_at import runner as random_runner

import numpy as np
import os

if __name__ == '__main__':
    # overall settings
    base_log_dir = './data'
    max_path_length = 4 # (FalStar MCTS used 6)
    s_0 = np.array([0])

    if os.environ.get("SEED") is not None:
        seed = os.environ["SEED"] # When running `run_algo.py` to get different seeds
    else:
        seed = 0
    print("Using seed: ", seed)

    # experiment settings
    run_experiment_args = {'snapshot_mode': 'gap',
                           'snapshot_gap': 1,
                           'log_dir': None,
                           'exp_name': None,
                           'seed': seed,
                           'n_parallel': 1,
                           'tabular_log_file': 'progress.csv',
                           'python_command': 'python3' # because /usr/bin/python is version 3.
                           }

    # runner settings
    runner_args = {'n_epochs': 1000,
                   'batch_size': 40,
                   'plot': False
                   }

    # env settings
    env_args = {'id': 'ast_toolbox:Autotrans_shift',
                'blackbox_sim_state': True,
                'open_loop': False,
                'fixed_init_state': True,
                's_0': s_0,
                }

    # simulation settings
    sim_args = {'blackbox_sim_state': True,
                'open_loop': False,
                'fixed_initial_state': True,
                'max_path_length': max_path_length
                }

    # reward settings
    reward_args = {}

    # spaces settings
    spaces_args = {}

    # DRL Settings
    drl_policy_args = {'name': 'lstm_policy',
                       'hidden_dim': 64,
                       'use_peepholes': True,
                       }
    drl_baseline_args = {}
    drl_algo_args = {'max_path_length': max_path_length,
                     'discount': 1.0,
                     'lr_clip_range': 1.0,
                     'max_kl_step': 1.0,
                     }

    # MCTS Settings
    mcts_policy_args = {}

    mcts_baseline_args = {}

    mcts_algo_args = {'max_path_length': max_path_length,
                      'stress_test_mode': 2,
                      'ec': 10.0,
                      'n_itr': runner_args['n_epochs'],
                      'k': 1,
                      'alpha': 0.7,
                      'clear_nodes': True,
                      'log_interval': max_path_length,
                      'plot_tree': False,
                      'plot_path': None,
                      'log_dir': None, # set below.
                     }

    mcts_bpq_args = {'N': 1}


    # Random search settings
    random_algo_args = {'max_path_length': max_path_length,
                        'discount': 1.0,
                       }
    random_bpq_args = {'N': 1}

    exp_log_dir = base_log_dir
    env_args['s_0'] = s_0
    sim_args['max_path_length'] = max_path_length

    # NOTE: os.environ.get("ALGO") when running from `run_algo.py`
    algo_from_env = os.environ.get("ALGO")
    if algo_from_env == None:
        algo2run = "drl" # default to run DRL
    else:
        algo2run = algo_from_env

    # DRL settings
    if algo2run == "drl":
        run_experiment_args['log_dir'] = exp_log_dir + '/drl'
        run_experiment_args['exp_name'] = 'drl'

        drl_algo_args['max_path_length'] = max_path_length

        drl_runner(
            env_args=env_args,
            run_experiment_args=run_experiment_args,
            sim_args=sim_args,
            reward_args=reward_args,
            spaces_args=spaces_args,
            algo_args=drl_algo_args,
            runner_args=runner_args,
        )

    # MCTS settings
    if algo2run == "mcts":
        run_experiment_args['log_dir'] = exp_log_dir + '/mcts'
        run_experiment_args['exp_name'] = 'mcts'

        mcts_algo_args['max_path_length'] = max_path_length
        mcts_algo_args['log_dir'] = run_experiment_args['log_dir']
        mcts_algo_args['plot_path'] = run_experiment_args['log_dir']

        mcts_runner(
            env_args=env_args,
            run_experiment_args=run_experiment_args,
            sim_args=sim_args,
            reward_args=reward_args,
            spaces_args=spaces_args,
            algo_args=mcts_algo_args,
            bpq_args=mcts_bpq_args,
            runner_args=runner_args,
        )

    # Random search settings
    if algo2run == "random":
        run_experiment_args['log_dir'] = exp_log_dir + '/random'
        run_experiment_args['exp_name'] = 'random'

        random_algo_args['max_path_length'] = max_path_length

        random_runner(
            env_args=env_args,
            run_experiment_args=run_experiment_args,
            sim_args=sim_args,
            reward_args=reward_args,
            spaces_args=spaces_args,
            algo_args=random_algo_args,
            bpq_args=random_bpq_args,
            runner_args=runner_args,
        )
