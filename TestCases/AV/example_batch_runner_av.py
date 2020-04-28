from TestCases.AV.example_runner_ge_av import runner as go_explore_runner
from TestCases.AV.example_runner_drl_av import runner as drl_runner
from TestCases.AV.example_runner_mcts_av import runner as mcts_runner
from TestCases.AV.example_runner_ba_av import runner as ba_runner
import pickle
import pdb

if __name__ == '__main__':
    # Overall settings
    max_path_length = 50
    s_0 = [0.0, -4.0, 1.0, 11.17, -35.0]
    base_log_dir = '/home/mkoren/scratch/data/dif_test'
    # experiment settings
    run_experiment_args = {'snapshot_mode':'last',
                           'snapshot_gap':1,
                           'log_dir':None,
                           'exp_name':None,
                           'seed':0,
                           'n_parallel':8,
                           'tabular_log_file':'progress.csv'
                           }

    # runner settings
    runner_args = {'n_epochs':100,
                   'batch_size':500,
                   'plot':False
                   }

    # env settings
    env_args = {'id':'mylab:GoExploreAST-v1',
                'blackbox_sim_state':True,
                'open_loop':False,
                'fixed_init_state':True,
                's_0':s_0,
                }

    # simulation settings
    sim_args = {'blackbox_sim_state':True,
                'open_loop':False,
                'fixed_initial_state':True,
                'max_path_length':max_path_length,
                'dt':0.1
                }

    # reward settings
    reward_args = {'use_heuristic':True}

    # spaces settings
    spaces_args = {}



    # Go-Explore Settings

    ge_algo_args = {'db_filename':None,
                    'max_db_size':150,
                    'max_path_length':max_path_length,
                    'discount':0.99,
                    'save_paths_gap':1,
                    'save_paths_path':None,
                    'overwrite_db':True,
                    'use_score_weight':True
                    }

    ge_baseline_args = {}

    ge_policy_args = {}

    mcts_policy_args = {}

    mcts_baseline_args = {}

    mcts_algo_args = {'max_path_length': max_path_length,
                      'stress_test_num': 2,
                      'ec': 100.0,
                      'n_itr': runner_args['n_epochs'] * runner_args['batch_size'] / max_path_length ** 2,
                      'k': 0.5,
                      'alpha': 0.5,
                      'gamma':0.99,
                      'clear_nodes': True,
                      'log_interval': 500,
                      'log_tabular': True,
                      'plot_tree': False,
                      'plot_path': None,
                      'log_dir': None,
                      }

    mcts_bpq_args = {'N': 10}

    exp_log_dir = base_log_dir
    dilation_factor = 2
    max_path_length = 50 * dilation_factor
    sim_args['dt'] = 0.1 / dilation_factor
    s_0 = [0.0, -6, 1.0, 11.17, -35.0]
    env_args['s_0'] = s_0
    reward_args['use_heuristic'] = False
    sim_args['max_path_length'] = max_path_length

    # Easy GE settings
    # run_experiment_args['log_dir'] = exp_log_dir + '/ge'
    # run_experiment_args['exp_name'] = 'ge'
    # # run_experiment_args['tabular_log_file'] = 'progress.csv'
    #
    # ge_algo_args['db_filename'] = run_experiment_args['log_dir'] + '/cellpool.dat'
    # ge_algo_args['save_paths_path'] = run_experiment_args['log_dir']
    # ge_algo_args['max_path_length'] = max_path_length
    # ge_algo_args['use_score_weight'] = True
    # # run_experiment_args['seed'] = 1
    # runner_args['batch_size'] = 500
    #
    # go_explore_runner(
    #     env_args=env_args,
    #     run_experiment_args=run_experiment_args,
    #     sim_args=sim_args,
    #     reward_args=reward_args,
    #     spaces_args=spaces_args,
    #     policy_args=ge_policy_args,
    #     baseline_args=ge_baseline_args,
    #     algo_args=ge_algo_args,
    #     runner_args=runner_args,
    #     # log_dir='.',
    # )

    # Hard MCTS settings
    # run_experiment_args['log_dir'] = exp_log_dir + '/mcts'
    # run_experiment_args['exp_name'] = 'mcts'
    #
    # mcts_algo_args['max_path_length'] = max_path_length
    # mcts_algo_args['log_dir'] = run_experiment_args['log_dir']
    # mcts_algo_args['plot_path'] = run_experiment_args['log_dir']
    # # mcts_algo_args['n_itr'] = runner_args['n_epochs']*runner_args['batch_size'] // max_path_length**2
    # mcts_algo_args['n_itr'] = 100
    # runner_args['batch_size'] = 500
    #
    # mcts_runner(
    #     env_args=env_args,
    #     run_experiment_args=run_experiment_args,
    #     sim_args=sim_args,
    #     reward_args=reward_args,
    #     spaces_args=spaces_args,
    #     policy_args=mcts_policy_args,
    #     baseline_args=mcts_baseline_args,
    #     algo_args=mcts_algo_args,
    #     bpq_args=mcts_bpq_args,
    #     runner_args=runner_args,
    #     # log_dir='.',
    # )