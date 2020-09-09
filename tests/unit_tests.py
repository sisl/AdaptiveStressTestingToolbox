from unittest.mock import patch

import numpy as np
import pytest
from bsddb3 import db
from gym.spaces.box import Box

import ast_toolbox.samplers.parallel_sampler as ps
from ast_toolbox.algos.go_explore import Cell
from ast_toolbox.algos.go_explore import CellPool
from ast_toolbox.envs.go_explore_ast_env import Custom_GoExploreASTEnv
from ast_toolbox.envs.go_explore_ast_env import GoExploreASTEnv
from ast_toolbox.envs.go_explore_ast_env import GoExploreParameter
from ast_toolbox.simulators import ASTSimulator
from ast_toolbox.simulators import ExampleAVSimulator
from ast_toolbox.spaces import ASTSpaces
from ast_toolbox.spaces import ExampleAVSpaces
from examples.AV.example_runner_ba_av import runner as ba_runner
from examples.AV.example_runner_drl_av import runner as drl_runner
from examples.AV.example_runner_ga_av import runner as ga_runner
from examples.AV.example_runner_ge_av import runner as ge_runner
from examples.AV.example_runner_mcts_av import runner as mcts_runner
from tests.validate_drl import validate_drl
from tests.validate_ga import validate_ga
from tests.validate_ge_ba import validate_ge_ba
from tests.validate_install import validate_install
from tests.validate_mcts import validate_mcts
from tests.validate_parallel import validate_parallel


def test_validate_install():
    assert validate_install() is True


def test_validate_parallel():
    assert validate_parallel() is True


def test_validate_drl():
    assert validate_drl() is True


def test_validate_mcts():
    assert validate_mcts() is True


def test_validate_ga():
    assert validate_ga() is True


def test_validate_ge_ba():
    assert validate_ge_ba() is True


def test_ast_simulator():
    sim = ASTSimulator()
    with pytest.raises(NotImplementedError):
        sim.simulate(None, None)
    sim.open_loop = False
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
    assert np.all(sim.reset(np.array([0, 0, 0, 0, 0])) == np.array([1, 1, 1, 1]))
    sim.open_loop = False
    assert np.all(sim.closed_loop_step(action=np.array([])) == np.array([1, 1, 1, 1]))
    sim.open_loop = True

    sim.blackbox_sim_state = True
    assert np.all(sim.observation_return() == np.array([0, 0, 0, 0, 0]))
    assert np.all(sim.reset(np.array([0, 0, 0, 0, 0])) == np.array([0, 0, 0, 0, 0]))
    sim.open_loop = False
    assert np.all(sim.closed_loop_step(action=np.array([])) == np.array([0, 0, 0, 0, 0]))
    sim.open_loop = True

    assert sim.log() is None


def test_example_av_simulator():
    simulator_args = {'car_init_x': 0,
                      'car_init_y': 0}
    sim = ExampleAVSimulator(simulator_args=simulator_args, max_path_length=1)
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


def test_ast_spaces():
    space = ASTSpaces()

    with pytest.raises(NotImplementedError):
        space.action_space()

    with pytest.raises(NotImplementedError):
        space.observation_space()


def test_example_av_spaces():
    space = ExampleAVSpaces(num_peds=2)

    assert isinstance(space.action_space, Box)
    assert isinstance(space.observation_space, Box)


def test_go_explore_ast_env():
    env = GoExploreASTEnv()
    env._fixed_init_state = True
    env._init_state = np.zeros(5)
    env.p_robustify_state = None
    # import pdb; pdb.set_trace()
    with patch('ast_toolbox.envs.go_explore_ast_env.db.DB', side_effect=db.DBBusyError):
        assert np.all(env.reset() == np.zeros(5))
    with patch('ast_toolbox.envs.go_explore_ast_env.db.DB', side_effect=db.DBLockNotGrantedError):
        assert np.all(env.reset() == np.zeros(5))
    with patch('ast_toolbox.envs.go_explore_ast_env.db.DB', side_effect=db.DBForeignConflictError):
        assert np.all(env.reset() == np.zeros(5))
    with patch('ast_toolbox.envs.go_explore_ast_env.db.DB', side_effect=db.DBAccessError):
        assert np.all(env.reset() == np.zeros(5))
    with patch('ast_toolbox.envs.go_explore_ast_env.db.DB', side_effect=db.DBPermissionsError):
        assert np.all(env.reset() == np.zeros(5))
    with patch('ast_toolbox.envs.go_explore_ast_env.db.DB', side_effect=db.DBNoSuchFileError):
        assert np.all(env.reset() == np.zeros(5))
    with patch('ast_toolbox.envs.go_explore_ast_env.db.DB', side_effect=db.DBError):
        assert np.all(env.reset() == np.zeros(5))

    env.p_key_list = GoExploreParameter(name='key_list', value=[0])
    env.p_max_value = GoExploreParameter(name='max_value', value=1)
    env.sample_limit = 10

    # env.p_key_list.value = [0]

    class Test_Pop:
        def __init__(self, fitness):
            self.fitness = fitness

    test_pop = Test_Pop(0.1)
    with patch('ast_toolbox.envs.go_explore_ast_env.random.random', return_value=1.0):
        assert env.sample(population=[test_pop]) == test_pop
    test_pop = Test_Pop(0)
    with patch('ast_toolbox.envs.go_explore_ast_env.random.random', return_value=1.0):
        with pytest.raises(ValueError):
            env.sample(population=[test_pop])

    env.simulator.blackbox_sim_state = True
    env.simulator.initial_conditions = np.zeros(5)
    # import pdb; pdb.set_trace()
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
    assert np.all(cenv.downsample(obs) == np.array([1, 0, 1, 1, 2]))

    assert isinstance(env.observation_space, Box)
    assert isinstance(env.action_space, Box)

    env._info = 'test'
    assert env.get_cache_list() == 'test'


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


def test_go_explore():
    cell1 = Cell(use_score_weight=False)
    cell2 = Cell(use_score_weight=False)
    cell1.observation = np.zeros(5)
    cell2.observation = np.zeros(5)

    assert cell1 != np.zeros(5)
    assert cell1 == cell2

    cell2.observation = np.ones(5)
    assert cell1 != cell2

    assert cell1.is_root
    assert cell1.step == 0

    cell1.fitness
    assert 'fitness' in cell1.__dict__
    cell1.times_visited = 1
    assert 'fitness' not in cell1.__dict__

    assert hash(cell1) == hash((cell1.observation.tostring()))

    cell_pool = CellPool(filename='./test_pool.dat', use_score_weight=True)
    d_pool = cell_pool.open_pool(overwrite=True)
    cell_pool.d_update(
        cell_pool_shelf=d_pool,
        observation=np.zeros(5),
        action=np.zeros(5),
        trajectory=np.array(
            []),
        score=0.0,
        state=None,
        reward=0.0,
        chosen=0)
    cell_pool.d_update(cell_pool_shelf=d_pool, observation=np.zeros(5), action=np.zeros(5), trajectory=np.array([]), score=1.0,
                       state=None, reward=1.0, chosen=0, is_goal=True)


def test_example_runner_ba_av():
    sim = ExampleAVSimulator()
    sim.reset(np.zeros(5))
    expert_trajectory = {'state': sim.clone_state(),
                         'reward': 0,
                         'action': np.zeros(6),
                         'observation': np.zeros(5)}
    with patch('examples.AV.example_runner_ba_av.compress_pickle.dump', side_effect=MemoryError):
        with patch('examples.AV.example_runner_ba_av.LocalTFRunner.train', new=lambda x: 0):
            ba_runner(
                env_args={
                    'id': 'ast_toolbox:GoExploreAST-v1'},
                algo_args={
                    'expert_trajectory': [expert_trajectory] *
                    50,
                    'max_epochs': 10})


def test_example_runner_drl_av():
    with patch('examples.AV.example_runner_drl_av.run_experiment'):
        drl_runner(save_expert_trajectory=False)
#     # Create mock data from last iteration
#     mock_steps = 3
#     mock_env_info = {'actions':np.zeros((mock_steps, 5)),
#                  'state':np.zeros((mock_steps, 5)),}
#     mock_path = {'rewards':np.zeros(mock_steps),
#                  'observations':np.zeros((mock_steps, 5)),
#                  'env_infos':mock_env_info}
#     mock_paths = [mock_path]
#     mock_last_iter_data = {'paths':mock_paths}
#
#
#     with patch('examples.AV.test_example_runner_drl_av.LocalTFRunner.train'):
#         drl_runner(save_expert_trajectory=False)
#         with patch('examples.AV.test_example_runner_drl_av.open'):
#             with patch('examples.AV.test_example_runner_drl_av.pickle.load', return_value = mock_last_iter_data):
#                 with patch('examples.AV.test_example_runner_drl_av.pickle.dump'):
#                     drl_runner(save_expert_trajectory=True)


def test_example_runner_ga_av():
    with patch('examples.AV.example_runner_ga_av.run_experiment'):
        ga_runner()


def test_example_runner_ge_av():
    with patch('examples.AV.example_runner_ge_av.run_experiment'):
        ge_runner()


def test_example_runner_mcts_av():
    with patch('examples.AV.example_runner_mcts_av.run_experiment'):
        mcts_runner()
