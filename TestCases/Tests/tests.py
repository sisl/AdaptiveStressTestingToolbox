import unittest

from TestCases.Tests.validate_install import validate_install

class TestMethods(unittest.TestCase):
    def test_add(self):
        self.assertEqual(validate_install(), True)


if __name__ == '__main__':
    unittest.main()