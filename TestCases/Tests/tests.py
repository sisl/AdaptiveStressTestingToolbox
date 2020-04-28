# import unittest
#
# from TestCases.Tests.validate_install import validate_install
#
# class TestMethods(unittest.TestCase):
#     def test_add(self):
#         self.assertEqual(validate_install(), True)
#
#
# if __name__ == '__main__':
#     unittest.main()
import pytest
import numpy as np
from TestCases.Tests.validate_install import validate_install


def test_validate_install():
    assert validate_install() is True

from mylab.simulators.ast_simulator import ASTSimulator

def test_ast_simulator():
    sim = ASTSimulator()
    with pytest.raises(NotImplementedError):
        sim.simulate(None, None)
    with pytest.raises(NotImplementedError):
        sim.step(None, None)
    with pytest.raises(NotImplementedError):
        sim.reset(None)
    with pytest.raises(NotImplementedError):
        sim.get_reward_info()
    with pytest.raises(NotImplementedError):
        sim.is_goal()

    assert sim.log() is None

from mylab.simulators.example_av_simulator import ExampleAVSimulator

def test_example_av_simulator():
    sim = ExampleAVSimulator(car_init_x=0, car_init_y=0, max_path_length=1)

#     Check if simulate can find a goal
    path_length, info = sim.simulate(actions=[np.zeros(6)], s_0=np.array([0,0,0,0,0]))
    assert path_length == 0
    # Check if simulate can return end of path
    path_length, info = sim.simulate(actions=[np.zeros(6)], s_0=np.array([5,5,0,0,0]))
    assert path_length == -1

    # Check open-loop sim step
    obs = sim.step(action=None, open_loop=True)
    assert obs is None

    # Check closed-loop sim step
    init = sim.reset(s_0=np.array([0,0,0,0,0]))
    assert np.all(init == np.array([0,0,0,0,0]))
    obs = sim.step(action=np.zeros(6), open_loop=False)
    print(obs)
    assert np.all(obs == np.array([0,0,0,0]))

    # check white-box reset
    sim.action_only = False
    init = sim.reset(s_0=np.array([0,0,0,0,0]))
    assert np.all(init == np.array([0, 0, 0, 0]))