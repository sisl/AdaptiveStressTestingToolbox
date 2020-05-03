import pytest
from unittest.mock import patch
import numpy as np

# from TestCases.Tests.validate_install import validate_install
# from TestCases.Tests.validate_parallel import validate_parallel
# from TestCases.Tests.validate_drl import validate_drl
# from TestCases.Tests.validate_mcts import validate_mcts
# from TestCases.Tests.validate_ge_ba import validate_ge_ba
#
#
# def test_validate_install():
#     assert validate_install() is True
#
#
# def test_validate_parallel():
#     assert validate_parallel() is True
#
#
# def test_validate_drl():
#     assert validate_drl() is True
#
#
# def test_validate_mcts():
#     assert validate_mcts() is True
#
#
# def test_validate_ge_ba():
#     assert validate_ge_ba() is True
#
#
# from mylab.simulators.ast_simulator import ASTSimulator
#
#
# def test_ast_simulator():
#     sim = ASTSimulator()
#     with pytest.raises(NotImplementedError):
#         sim.simulate(None, None)
#     with pytest.raises(NotImplementedError):
#         sim.closed_loop_step(None)
#     sim.open_loop = False
#     with pytest.raises(NotImplementedError):
#         sim.step(None)
#     with pytest.raises(NotImplementedError):
#         sim.reset(None)
#     with pytest.raises(NotImplementedError):
#         sim.get_reward_info()
#     with pytest.raises(NotImplementedError):
#         sim.is_goal()
#
#     sim.open_loop = True
#     sim.initial_conditions = np.array([0, 0, 0, 0, 0])
#     sim.observation = np.array([1, 1, 1, 1])
#
#     sim.c_max_path_length = 2
#     sim._path_length = 0
#
#     assert np.all(sim.step(None) == np.array([0, 0, 0, 0, 0]))
#     assert (sim.is_terminal() is False)
#
#     assert np.all(sim.step(None) == np.array([0, 0, 0, 0, 0]))
#     assert (sim.is_terminal() is True)
#
#     sim.blackbox_sim_state = False
#     assert np.all(sim.observation_return() == np.array([1, 1, 1, 1]))
#
#     sim.blackbox_sim_state = True
#     assert np.all(sim.observation_return() == np.array([0, 0, 0, 0, 0]))
#
#     assert sim.log() is None
#
#
# from mylab.simulators.example_av_simulator import ExampleAVSimulator
#
#
# def test_example_av_simulator():
#     sim = ExampleAVSimulator(car_init_x=0, car_init_y=0, max_path_length=1)
#     sim.blackbox_sim_state = False
#
#     # check reset
#     init = sim.reset(s_0=np.array([0, 0, 0, 0, 0]))
#     assert np.all(init == np.array([0, 0, 0, 0]))
#
#     # Check if simulate can find a goal
#     path_length, info = sim.simulate(actions=[np.zeros(6)], s_0=np.array([0, 0, 0, 1, 0]))
#     assert path_length == 0
#     # Check if simulate can return end of path
#     path_length, info = sim.simulate(actions=[np.zeros(6)], s_0=np.array([5, 5, 0, 0, 0]))
#     assert path_length == -1
#
#     # Check open-loop sim step
#     sim.open_loop = True
#     sim.reset(s_0=np.array([0, 0, 0, 0, 0]))
#     obs = sim.step(action=np.zeros(6))
#     assert np.all(obs == np.array([0, 0, 0, 0, 0]))
#
#     # Check closed-loop sim step
#     sim.open_loop = False
#     init = sim.reset(s_0=np.array([0, 0, 0, 0, 0]))
#     obs = sim.step(action=np.zeros(6))
#     assert np.all(obs == np.array([0, 0, 0, 0]))
#
#
# from mylab.spaces.ast_spaces import ASTSpaces
#
#
# def test_ast_spaces():
#     space = ASTSpaces()
#
#     with pytest.raises(NotImplementedError):
#         space.action_space()
#
#     with pytest.raises(NotImplementedError):
#         space.observation_space()
#
#
# from mylab.spaces.example_av_spaces import ExampleAVSpaces
# from gym.spaces.box import Box
#
#
# def test_example_av_spaces():
#     space = ExampleAVSpaces(num_peds=2)
#
#     assert type(space.action_space) is Box
#     assert type(space.observation_space) is Box

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
    # env.sample_limit = 10
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

