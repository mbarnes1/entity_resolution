import unittest
import numpy as np
from experiments import SyntheticExperiment
from new_metrics import NewMetrics
__author__ = 'mbarnes1'


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Expensive setup items, only done once
        """
        experiment = SyntheticExperiment(2, 50)
        corruption_index = np.random.randint(0, len(experiment.corruption_multipliers))
        threshold_index = np.random.randint(0, len(experiment.thresholds))
        cls._new_metric = NewMetrics(experiment.synthetic_test[corruption_index].database,
                                     experiment.er[corruption_index][threshold_index])

    def test_get_records(self):
        """
        Tests whether _get_records returns the correct records for a random corruption and threshold
        """
        entity_index = np.random.randint(0, len(self._new_metric.er.entities))
        records = self._new_metric._get_records(entity_index)
        indices = set()
        for record in records:
            indices.update(record.line_indices)
        self.assertEqual(indices, set(self._new_metric.er.entities[entity_index].line_indices))

    def test_random_path(self):
        """
        Tests whether _random_path is the correct length
        """
        entity_index = np.random.randint(0, len(self._new_metric.er.entities))
        path_cost = self._new_metric._random_path(entity_index)
        self.assertEqual(len(path_cost), len(self._new_metric.er.entities[entity_index].line_indices)-1)
                         # path length = number records - 1

    def test_net_expected_cost(self):
        """
        Simply tests whether net_expected_cost runs
        """
        net_path_cost = self._new_metric.net_expected_cost
        print 'Net path cost:', net_path_cost


if __name__ == '__main__':
    unittest.main()