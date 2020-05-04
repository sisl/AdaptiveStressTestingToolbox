import pytest
from unittest.mock import patch
import numpy as np
import tensorflow as tf

from TestCases.Tests.validate_install import validate_install
from TestCases.Tests.validate_parallel import validate_parallel
from TestCases.Tests.validate_drl import validate_drl
from TestCases.Tests.validate_mcts import validate_mcts
from TestCases.Tests.validate_ge_ba import validate_ge_ba


def test_validate_install():
    assert validate_install() is True


def test_validate_parallel():
    assert validate_parallel() is True


def test_validate_drl():
    assert validate_drl() is True


def test_validate_mcts():
    assert validate_mcts() is True


def test_validate_ge_ba():
    assert validate_ge_ba() is True


from mylab.simulators.ast_simulator import ASTSimulator


def test_ast_simulator():
    sim = ASTSimulator()
    with pytest.raises(NotImplementedError):
        sim.simulate(None, None)
    with pytest.raises(NotImplementedError):
        sim.closed_loop_step(None)
    sim.open_loop = False
    with pytest.raises(NotImplementedError):
        sim.step(None)
    with pytest.raises(NotImplementedError):
        sim.reset(None)
    with pytest.raises(NotImplementedError):
        sim.get_reward_info()
    with pytest.raises(NotImplementedError):
        sim.is_goal()

    sim.open_loop = True
    sim.initial_conditions = np.array([0, 0, 0, 0, 0])
    sim.observation = np.array([1, 1, 1, 1])

    sim.c_max_path_length = 2
    sim._path_length = 0

    assert np.all(sim.step(None) == np.array([0, 0, 0, 0, 0]))
    assert (sim.is_terminal() is False)

    assert np.all(sim.step(None) == np.array([0, 0, 0, 0, 0]))
    assert (sim.is_terminal() is True)

    sim.blackbox_sim_state = False
    assert np.all(sim.observation_return() == np.array([1, 1, 1, 1]))

    sim.blackbox_sim_state = True
    assert np.all(sim.observation_return() == np.array([0, 0, 0, 0, 0]))

    assert sim.log() is None


from mylab.simulators.example_av_simulator import ExampleAVSimulator


def test_example_av_simulator():
    sim = ExampleAVSimulator(car_init_x=0, car_init_y=0, max_path_length=1)
    sim.blackbox_sim_state = False

    # check reset
    init = sim.reset(s_0=np.array([0, 0, 0, 0, 0]))
    assert np.all(init == np.array([0, 0, 0, 0]))

    # Check if simulate can find a goal
    path_length, info = sim.simulate(actions=[np.zeros(6)], s_0=np.array([0, 0, 0, 1, 0]))
    assert path_length == 0
    # Check if simulate can return end of path
    path_length, info = sim.simulate(actions=[np.zeros(6)], s_0=np.array([5, 5, 0, 0, 0]))
    assert path_length == -1

    # Check open-loop sim step
    sim.open_loop = True
    sim.reset(s_0=np.array([0, 0, 0, 0, 0]))
    obs = sim.step(action=np.zeros(6))
    assert np.all(obs == np.array([0, 0, 0, 0, 0]))

    # Check closed-loop sim step
    sim.open_loop = False
    init = sim.reset(s_0=np.array([0, 0, 0, 0, 0]))
    obs = sim.step(action=np.zeros(6))
    assert np.all(obs == np.array([0, 0, 0, 0]))


from mylab.spaces.ast_spaces import ASTSpaces


def test_ast_spaces():
    space = ASTSpaces()

    with pytest.raises(NotImplementedError):
        space.action_space()

    with pytest.raises(NotImplementedError):
        space.observation_space()


from mylab.spaces.example_av_spaces import ExampleAVSpaces
from gym.spaces.box import Box


def test_example_av_spaces():
    space = ExampleAVSpaces(num_peds=2)

    assert type(space.action_space) is Box
    assert type(space.observation_space) is Box

from mylab.envs.go_explore_ast_env import GoExploreASTEnv, GoExploreParameter, Custom_GoExploreASTEnv
from bsddb3 import db

def test_go_explore_ast_env():
    env = GoExploreASTEnv()
    env._fixed_init_state = True
    env._init_state = np.zeros(5)
    env.p_robustify_state = None
    # import pdb; pdb.set_trace()
    with patch('mylab.envs.go_explore_ast_env.db.DB', side_effect=db.DBBusyError):
        assert np.all(env.reset() == np.zeros(5))
    with patch('mylab.envs.go_explore_ast_env.db.DB', side_effect=db.DBLockNotGrantedError):
        assert np.all(env.reset() == np.zeros(5))
    with patch('mylab.envs.go_explore_ast_env.db.DB', side_effect=db.DBForeignConflictError):
        assert np.all(env.reset() == np.zeros(5))
    with patch('mylab.envs.go_explore_ast_env.db.DB', side_effect=db.DBAccessError):
        assert np.all(env.reset() == np.zeros(5))
    with patch('mylab.envs.go_explore_ast_env.db.DB', side_effect=db.DBPermissionsError):
        assert np.all(env.reset() == np.zeros(5))
    with patch('mylab.envs.go_explore_ast_env.db.DB', side_effect=db.DBNoSuchFileError):
        assert np.all(env.reset() == np.zeros(5))
    with patch('mylab.envs.go_explore_ast_env.db.DB', side_effect=db.DBError):
        assert np.all(env.reset() == np.zeros(5))


    env.p_key_list = GoExploreParameter(name='key_list', value=[0])
    env.p_max_value = GoExploreParameter(name='max_value', value=1)
    env.sample_limit = 10
    # env.p_key_list.value = [0]
    class Test_Pop:
        def __init__(self, fitness):
            self.fitness = fitness

    test_pop = Test_Pop(0.1)
    with patch('mylab.envs.go_explore_ast_env.random.random', return_value=1.0):
        assert env.sample(population=[test_pop]) == test_pop
    test_pop = Test_Pop(0)
    with patch('mylab.envs.go_explore_ast_env.random.random', return_value=1.0):
        with pytest.raises(ValueError):
            env.sample(population=[test_pop])

    env.simulator.blackbox_sim_state = True
    env.simulator.initial_conditions = np.zeros(5)
    # import pdb; pdb.set_trace()
    assert np.all(env._get_obs() == np.zeros(5))
    env.render(car=None, ped=None, noise=None)

    env._init_state = np.array([0, 0, 0, 1, 0])
    env.simulator.c_max_path_length = 1
    path_length, info = env.simulate(actions=[np.zeros(6)])
    assert path_length == 0
    env._fixed_init_state = False
    path_length, info = env.simulate(actions=[np.zeros(6)])
    assert path_length == -1

    env.blackbox_sim_state = True
    env.simulator.blackbox_sim_state = True
    obs = env.env_reset()
    assert env.observation_space.contains(obs)

    env.blackbox_sim_state = False
    env.simulator.blackbox_sim_state = False
    obs = env.env_reset()
    assert env.observation_space.contains(obs[4:])

    obs = np.array([0, 0.001, 0.0015, 0.002])
    assert np.all(env.downsample(obs) == obs)



    cenv = Custom_GoExploreASTEnv()
    cenv._step = 1
    assert np.all(cenv.downsample(obs) == np.array([1,0,1,1,2]))


import mylab.samplers.parallel_sampler as ps
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicy
import gym


def test_parallel_sampler():
    # env = TfEnv(env_name='CartPole-v1')
    #
    # policy = CategoricalMLPPolicy(
    #     name='policy', env_spec=env.spec, hidden_sizes=(32, 32))
    env = None
    policy = None
    ps.initialize(1)
    ps.set_seed(0)
    ps.populate_task(env, policy)
    ps.close()

    ps.initialize(2)
    ps.set_seed(0)
    ps.populate_task(env, policy, scope=0)
    ps.populate_task(env, policy, scope=1)
    ps.close()

from mylab.samplers.batch_sampler import BatchSampler
from garage.sampler import singleton_pool
from mylab.envs.go_explore_ast_env import GoExploreASTEnv
from garage.tf.algos.ppo import PPO
from garage.tf.envs.base import TfEnv
from garage.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
from garage.tf.policies.uniform_control_policy import UniformControlPolicy
from garage.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from garage.np.baselines.linear_feature_baseline import LinearFeatureBaseline
from garage.envs.normalized_env import normalize
from mylab.simulators.example_av_simulator import ExampleAVSimulator
from mylab.rewards.example_av_reward import ExampleAVReward
from mylab.spaces.example_av_spaces import ExampleAVSpaces

# # Import the AST classes
# from mylab.envs.ast_env import ASTEnv
# from mylab.samplers.ast_vectorized_sampler import ASTVectorizedSampler
# def test_batch_sampler():
#
#     # env settings
#     s_0 = [0.0, -4.0, 1.0, 11.17, -35.0]
#     env_args = {'id':'mylab:GoExploreAST-v1',
#                 'blackbox_sim_state':True,
#                 'open_loop':False,
#                 'fixed_init_state':True,
#                 's_0':s_0,
#                 }
#
#     # simulation settings
#     sim_args = {'blackbox_sim_state':True,
#                 'open_loop':False,
#                 'fixed_initial_state':True,
#                 'max_path_length':max_path_length
#                 }
#
#     # reward settings
#     reward_args = {'use_heuristic':True}
#
#     # spaces settings
#     spaces_args = {}
#
#     # DRL Settings
#
#     drl_policy_args = {'name':'lstm_policy',
#                        'hidden_dim':64,
#                        'use_peepholes':True,
#                        }
#
#     drl_baseline_args = {}
#
#     drl_algo_args = {'max_path_length':5,
#                      'discount':0.99,
#                      'lr_clip_range':1.0,
#                      'max_kl_step':1.0,
#                      # 'log_dir':None,
#                      }
#
#     sim = ExampleAVSimulator(**sim_args)
#     reward_function = ExampleAVReward(**reward_args)
#     spaces = ExampleAVSpaces(**spaces_args)
#
#     # Create the environment
#     env = TfEnv(normalize(ASTEnv(simulator=sim,
#                                  reward_function=reward_function,
#                                  spaces=spaces,
#                                  **env_args,
#                                  )))
#
#     # Instantiate the garage objects
#     policy = GaussianLSTMPolicy(env_spec=env.spec, **drl_policy_args)
#     # name='lstm_policy',
#     # env_spec=env.spec,
#     # hidden_dim=64,
#     # 'use_peepholes=True)
#
#     baseline = LinearFeatureBaseline(env_spec=env.spec, **drl_baseline_args)
#
#     optimizer = ConjugateGradientOptimizer
#     optimizer_args = {'hvp_approach': FiniteDifferenceHvp(base_eps=1e-5)}
#
#     algo = PPO(env_spec=env.spec,
#                policy=policy,
#                baseline=baseline,
#                optimizer=optimizer,
#                optimizer_args=optimizer_args,
#                **drl_algo_args
#                )
#
#     singleton_pool.initialize(2)
#     sess = tf.compat.v1.Session()
#     sess.__enter__()
#     bs = BatchSampler(algo=algo, env=env)
#
#     with tf.name_scope('initialize_tf_vars'):
#         uninited_set = [
#             e.decode() for e in sess.run(
#                 tf.compat.v1.report_uninitialized_variables())
#         ]
#         sess.run(
#             tf.compat.v1.variables_initializer([
#                 v for v in tf.compat.v1.global_variables()
#                 if v.name.split(':')[0] in uninited_set
#             ]))
#     bs.start_worker()
#     bs.shutdown_worker()

