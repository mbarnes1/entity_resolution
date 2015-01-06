import unittest
import sys
from experiments import SyntheticExperiment
__author__ = 'mbarnes1'


class MyTestCase(unittest.TestCase):
    def setUp(self):
        # Create synthetic database, run ER on it
        self.experiment = SyntheticExperiment(2, 50)

if __name__ == '__main__':
    unittest.main()