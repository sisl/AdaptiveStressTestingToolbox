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
