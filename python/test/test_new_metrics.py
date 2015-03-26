import unittest
import numpy as np
from experiments import SyntheticExperiment, count_pairwise_class_balance
from new_metrics import NewMetrics
__author__ = 'mbarnes1'


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Expensive setup items, only done once
        """
        experiment = SyntheticExperiment(2, 50, 30, 30, 0.5, 3)
        cls.er = experiment.er
        cls.corruption_index = np.random.randint(0, len(experiment.corruption_multipliers))
        cls.threshold_index = np.random.randint(0, len(experiment.thresholds))
        class_balance_test = count_pairwise_class_balance(experiment.synthetic_test[cls.corruption_index].labels)
        cls._new_metric = NewMetrics(experiment.synthetic_test[cls.corruption_index].database,
                                     experiment.synthetic_test[cls.corruption_index].labels,
                                     experiment.er[cls.corruption_index][cls.threshold_index], class_balance_test)

    def test_pairwise_bounds(self):
        """
        Tests pairwise recall and precision lower bounds
        """

if __name__ == '__main__':
    unittest.main()