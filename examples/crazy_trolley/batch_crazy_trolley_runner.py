import pickle

from examples.AV.example_runner_ba_av import runner as ba_runner
from examples.AV.example_runner_drl_av import runner as drl_runner
from examples.AV.example_runner_ge_av import runner as go_explore_runner
from examples.crazy_trolley.mcts_crazy_trolley_runner import runner as mcts_runner

if __name__ == '__main__':
    # Which algorithms to run
    RUN_DRL = False
    RUN_MCTS = True
    RUN_GA = False
    RUN_GE = False
    RUN_BA = False

    # Overall settings
    max_path_length = 10
    s_0 = [0]
    base_log_dir = './data'
    # experiment settings
    run_experiment_args = {'snapshot_mode': 'last',
                           'snapshot_gap': 1,
                           'log_dir': None,
                           'exp_name': None,
                           'seed': 0,
                           'n_parallel': 10,
                           'tabular_log_file': 'progress.csv'
                           }

    # runner settings
    runner_args = {'n_epochs': 1,
                   'batch_size': 100,
                   'plot': False
                   }

    # env settings
    env_args = {'blackbox_sim_state': True,
                'open_loop': False,
                'fixed_init_state': True,
                's_0': s_0,
                }

    # simulation settings
    sim_args = {
                'blackbox_sim_state': True,
                'open_loop': False,
                'fixed_initial_state': True,
                'max_path_length': max_path_length,
                'height': 84,
                'width': 84,
                'from_pixels': True,
                'rgb': True,
                'random_level': False,
                'skip': 5,
                'noop': 30,
                'stack_frames': 5,
                'max_and_skip': True,
                'grayscale': True,
                'resize': None
                }

    # reward settings
    reward_args = {'use_heuristic': False}

    # spaces settings
    spaces_args = {'max_seed': 1e6}

    # MCTS ----------------------------------------------------------------------------------

    if RUN_MCTS:
        # MCTS Settings

        mcts_type = 'mcts'

        mcts_sampler_args = {}

        mcts_algo_args = {'max_path_length': max_path_length,
                          'stress_test_mode': 2,
                          'ec': 100.0,
                          'n_itr': 1,
                          'k': 0.5,
                          'alpha': 0.5,
                          'clear_nodes': True,
                          'log_interval': 50,
                          'plot_tree': False,
                          'plot_path': None,
                          'log_dir': None,
                          }

        mcts_bpq_args = {'N': 10}

        # MCTS settings
        run_experiment_args['log_dir'] = base_log_dir + '/mcts'
        run_experiment_args['exp_name'] = 'mcts'

        mcts_algo_args['max_path_length'] = max_path_length
        mcts_algo_args['log_dir'] = run_experiment_args['log_dir']
        mcts_algo_args['plot_path'] = run_experiment_args['log_dir']

        mcts_runner(
            mcts_type=mcts_type,
            env_args=env_args,
            run_experiment_args=run_experiment_args,
            sim_args=sim_args,
            reward_args=reward_args,
            spaces_args=spaces_args,
            algo_args=mcts_algo_args,
            runner_args=runner_args,
            bpq_args=mcts_bpq_args,
            sampler_args=mcts_sampler_args,
            save_expert_trajectory=True,
        )



