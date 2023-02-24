import unittest
from tests.test_perturbation import PerturbationTestCase

def suite():
    suite = unittest.TestSuite()
    suite.addTest(PerturbationTestCase())
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())