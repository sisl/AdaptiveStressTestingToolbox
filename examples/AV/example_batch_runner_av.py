import pickle

from examples.AV.example_runner_ba_av import runner as ba_runner
from examples.AV.example_runner_drl_av import runner as drl_runner
from examples.AV.example_runner_ge_av import runner as go_explore_runner
from examples.AV.example_runner_mcts_av import runner as mcts_runner

if __name__ == '__main__':
    # Which algorithms to run
    RUN_DRL = False
    RUN_MCTS = True
    RUN_GA = False
    RUN_GE = False
    RUN_BA = False

    # Overall settings
    max_path_length = 50
    s_0 = [0.0, -4.0, 1.0, 11.17, -35.0]
    base_log_dir = './data'
    # experiment settings
    run_experiment_args = {'snapshot_mode': 'last',
                           'snapshot_gap': 1,
                           'log_dir': None,
                           'exp_name': None,
                           'seed': 0,
                           'n_parallel': 8,
                           'tabular_log_file': 'progress.csv'
                           }

    # runner settings
    runner_args = {'n_epochs': 5,
                   'batch_size': 500,
                   'plot': False
                   }

    # env settings
    env_args = {'id': 'ast_toolbox:GoExploreAST-v1',
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
    reward_args = {'use_heuristic': True}

    # spaces settings
    spaces_args = {}

    # DRL ----------------------------------------------------------------------------------

    if RUN_DRL:
        # DRL Settings

        drl_policy_args = {'name': 'lstm_policy',
                           'hidden_dim': 64,
                           }

        drl_baseline_args = {}

        drl_algo_args = {'max_path_length': max_path_length,
                         'discount': 0.99,
                         'lr_clip_range': 1.0,
                         'max_kl_step': 1.0,
                         # 'log_dir':None,
                         }

        run_experiment_args['log_dir'] = base_log_dir + '/drl'
        run_experiment_args['exp_name'] = 'drl'

        drl_algo_args['max_path_length'] = max_path_length

        # Run DRL
        drl_runner(
            env_args=env_args,
            run_experiment_args=run_experiment_args,
            sim_args=sim_args,
            reward_args=reward_args,
            spaces_args=spaces_args,
            policy_args=drl_policy_args,
            baseline_args=drl_baseline_args,
            algo_args=drl_algo_args,
            runner_args=runner_args,
        )

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
                          'log_interval': 500,
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

    # Go-Explore ----------------------------------------------------------------------------------

    if RUN_GE:
        # Go-Explore Settings

        ge_algo_args = {'db_filename': None,
                        'max_db_size': 150,
                        'max_path_length': max_path_length,
                        'discount': 0.99,
                        'save_paths_gap': 1,
                        'save_paths_path': None,
                        'overwrite_db': True,
                        }

        ge_baseline_args = {}

        ge_policy_args = {}

        run_experiment_args['log_dir'] = base_log_dir + '/ge'
        run_experiment_args['exp_name'] = 'ge'
        # run_experiment_args['tabular_log_file'] = 'progress.csv'

        ge_algo_args['db_filename'] = run_experiment_args['log_dir'] + '/cellpool'
        ge_algo_args['save_paths_path'] = run_experiment_args['log_dir']
        ge_algo_args['max_path_length'] = max_path_length

        go_explore_runner(
            env_args=env_args,
            run_experiment_args=run_experiment_args,
            sim_args=sim_args,
            reward_args=reward_args,
            spaces_args=spaces_args,
            policy_args=ge_policy_args,
            baseline_args=ge_baseline_args,
            algo_args=ge_algo_args,
            runner_args=runner_args,
        )

    # Backward Algorithm ----------------------------------------------------------------------------------

    if RUN_BA:
        # BA Settings
        ba_algo_args = {'expert_trajectory': None,
                        'max_path_length': max_path_length,
                        'epochs_per_step': 10,
                        'scope': None,
                        'discount': 0.99,
                        'gae_lambda': 1.0,
                        'center_adv': True,
                        'positive_adv': False,
                        'fixed_horizon': False,
                        'pg_loss': 'surrogate_clip',
                        'lr_clip_range': 1.0,
                        'max_kl_step': 1.0,
                        'policy_ent_coeff': 0.0,
                        'use_softplus_entropy': False,
                        'use_neg_logli_entropy': False,
                        'stop_entropy_gradient': False,
                        'entropy_method': 'no_entropy',
                        'name': 'PPO',
                        'log_dir': None,
                        }

        ba_baseline_args = {}

        ba_policy_args = {'name': 'lstm_policy',
                          'hidden_dim': 64,
                          }

        with open(run_experiment_args['log_dir'] + '/expert_trajectory.p', 'rb') as f:
            expert_trajectories = pickle.load(f)

        run_experiment_args['log_dir'] = run_experiment_args['log_dir'] + '/ba'
        ba_algo_args['expert_trajectory'] = expert_trajectories[-1]

        ba_runner(
            env_args=env_args,
            run_experiment_args=run_experiment_args,
            sim_args=sim_args,
            reward_args=reward_args,
            spaces_args=spaces_args,
            policy_args=ba_policy_args,
            baseline_args=ba_baseline_args,
            algo_args=ba_algo_args,
            runner_args=runner_args,
        )
