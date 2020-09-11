import pickle

from examples.AV.example_runner_ba_av import runner as ba_runner
from examples.AV.example_runner_ge_av import runner as go_explore_runner


def validate_ge_ba():
    # Overall settings
    max_path_length = 5
    s_0 = [0.0, -4.0, 1.0, 11.17, -35.0]
    base_log_dir = './data'
    # experiment settings
    run_experiment_args = {'snapshot_mode': 'last',
                           'snapshot_gap': 1,
                           'log_dir': None,
                           'exp_name': None,
                           'seed': 0,
                           'n_parallel': 1,
                           'tabular_log_file': 'progress.csv'
                           }

    # runner settings
    runner_args = {'n_epochs': 1,
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

    sampler_args = {'n_envs': 1,
                    'open_loop': False}

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
    # BA Settings

    ba_algo_args = {'expert_trajectory': None,
                    'max_path_length': max_path_length,
                    'epochs_per_step': 1,
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
                    }

    ba_baseline_args = {}

    ba_policy_args = {'name': 'lstm_policy',
                      'hidden_dim': 64,
                      }

    exp_log_dir = base_log_dir
    env_args['s_0'] = s_0
    reward_args['use_heuristic'] = True
    sim_args['max_path_length'] = max_path_length
    ba_algo_args['max_path_length'] = max_path_length

    # GE settings
    run_experiment_args['log_dir'] = exp_log_dir + '/ge'
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
        sampler_args=sampler_args
    )

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
        sampler_args=sampler_args
    )

    return True


if __name__ == '__main__':
    validate_ge_ba()
