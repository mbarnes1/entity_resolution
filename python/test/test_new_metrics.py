import unittest
import numpy as np
from experiments import SyntheticExperiment
from new_metrics import NewMetrics, _min_dict, _max_dict
__author__ = 'mbarnes1'


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Expensive setup items, only done once
        """
        experiment = SyntheticExperiment(2, 50, 3)
        cls.corruption_index = np.random.randint(0, len(experiment.corruption_multipliers))
        cls.threshold_index = np.random.randint(0, len(experiment.thresholds))
        cls._new_metric = NewMetrics(experiment.synthetic_test[cls.corruption_index].database,
                                     experiment.er[cls.corruption_index][cls.threshold_index])


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

    def test_path_cost(self):
        """
        Simply tests whether net_expected_cost and greedy ran
        """
        expected_path_cost = self._new_metric.net_expected_cost
        print 'Net path cost:', expected_path_cost
        greedy_min_cost = self._new_metric.get_net_greedy_cost('best')
        greedy_max_cost = self._new_metric.get_net_greedy_cost('worst')
        print 'Greedy best path cost:', greedy_min_cost
        print 'Greedy worst path cost:', greedy_max_cost
        self.assertGreaterEqual(greedy_max_cost, expected_path_cost)
        self.assertGreaterEqual(expected_path_cost, greedy_min_cost)

    def test_min_max_dict(self):
        """
        Tests whether the min/max dictionary searches work correctly
        """
        dict0 = {
            frozenset(['a1', 'a2']): 0,
            frozenset(['b1', 'b2']): 1,
            frozenset(['c1', 'c2']): 2,
            frozenset(['d1', 'd2']): 3
        }
        self.assertEqual(_min_dict(dict0), frozenset(['a1', 'a2']))
        self.assertEqual(_max_dict(dict0), frozenset(['d1', 'd2']))
        self.assertEqual(_min_dict(dict0, remove={'a1'}), frozenset(['b1', 'b2']))
        dict1 = {
            frozenset(['b1', 'b2']): 1,
            frozenset(['c1', 'c2']): 2,
            frozenset(['d1', 'd2']): 3
        }
        self.assertEqual(dict0, dict1)
        self.assertEqual(_max_dict(dict0, remove={'b2', 'd1'}), frozenset(['c1', 'c2']))
        dict2 = {
            frozenset(['c1', 'c2']): 2
        }
        self.assertEqual(dict0, dict2)

    def test_greedy_cost(self):
        """
        Tests whether the greedy output makes sense
        """
        min_greedy_cost = self._new_metric.get_net_greedy_cost('best')
        max_greedy_cost = self._new_metric.get_net_greedy_cost('worst')




if __name__ == '__main__':
    unittest.main()