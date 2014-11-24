import unittest
from copy import deepcopy
from database import Synthetic, Database, remove_indices, find_in_list
import os
import numpy as np
__author__ = 'mbarnes1'


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self._test_path = 'test_annotations_cleaned.csv'

    def test_remove_indices(self):
        lst = ['one', '', 'two', '', 'three', '']
        new_list = remove_indices([1, 3, 5], lst)
        self.assertEqual(new_list, ['one', 'two', 'three'])

    def test_find_empty_entries(self):
        lst = ['one', '', 'two', '', 'three', '']
        empty_entries = find_in_list(lst, '')
        self.assertEqual([1, 3, 5], empty_entries)

    def test_sample_and_remove(self):
        database = Database(self._test_path)
        new_database = database.sample_and_remove(2)
        self.assertEqual(len(database.records), 2)
        self.assertEqual(len(new_database.records), 2)
        self.assertEqual(database.feature_descriptor, new_database.feature_descriptor)
        line_indices = set(database.records.keys()) | set(new_database.records.keys())
        self.assertEqual(line_indices, {0, 1, 2, 3})

    def test_size(self):
        database = Database(self._test_path)
        self.assertEqual(len(database.records), 4)

    def test_max_size(self):
        database = Database('test_annotations_10000_cleaned.csv', 409)
        self.assertEqual(len(database.records), 409)

    def test_synthetic(self):
        synthetic = Synthetic(10, 10, number_features=10, sigma=0)  # 100 records, 10 entities, 10 records each
        self.assertEqual(len(synthetic.labels), 100)
        self.assertEqual(synthetic.labels[0], 0)
        self.assertEqual(synthetic.labels[99], 9)
        self.assertEqual(len(synthetic.database.records), 100)
        self.assertEqual(synthetic.database.records[0].features, synthetic.database.records[1].features)  # line indices won't (and shouldn't) match. Different records
        self.assertEqual(synthetic.database.records[98].features, synthetic.database.records[98].features)
        self.assertNotEqual(synthetic.database.records[9].features, synthetic.database.records[10].features)
        self.assertEqual(synthetic.database.feature_descriptor.number, 10)

    def test_synthetic_number_features(self):
        synthetic = Synthetic(10, 10, number_features=2)
        self.assertEqual(synthetic.database.feature_descriptor.names, ['Name_0', 'Name_1'])
        self.assertEqual(synthetic.database.feature_descriptor.types, ['float', 'float'])
        self.assertEqual(synthetic.database.feature_descriptor.strengths, ['weak', 'weak'])
        self.assertEqual(synthetic.database.feature_descriptor.blocking, ['', ''])
        self.assertEqual(synthetic.database.feature_descriptor.pairwise_uses, ['numerical_difference', 'numerical_difference'])
        self.assertEqual(len(synthetic.database.records[0].features), 2)

    def test_synthetic_sample_and_remove(self):
        synthetic = Synthetic(10, 10, sigma=0)  # 100 records, 10 entities, 10 records each
        synthetic2 = synthetic.sample_and_remove(50)
        self.assertEqual(len(synthetic.database.records), 50)
        self.assertEqual(len(synthetic2.database.records), 50)
        self.assertEqual(set(synthetic.database.records.keys()), set(synthetic.labels.keys()))
        self.assertEqual(set(synthetic2.database.records.keys()), set(synthetic2.labels.keys()))
        self.assertEqual(len(set(synthetic.database.records.keys()) & set(synthetic2.database.records.keys())), 0)
        self.assertEqual(len(set(synthetic.labels.keys()) & set(synthetic2.labels.keys())), 0)

    def test_dump(self):
        database = Database(self._test_path)
        database.dump('test_annotations_dump.csv')
        database2 = Database('test_annotations_dump.csv')
        os.remove('test_annotations_dump.csv')
        self.assertEqual(database, database2)

    def test_plot(self):
        synthetic_original = Synthetic(10, 5, sigma=0)
        synthetic_original.plot(synthetic_original.labels, title='No Noise')
        synthetic_noise = deepcopy(synthetic_original)
        synthetic_noise.corrupt_evenly(0.1)
        synthetic_noise.plot(synthetic_noise.labels, title='Even Noise')
        synthetic_noise = deepcopy(synthetic_original)
        synthetic_noise.corrupt_clusters(0.2)
        synthetic_noise.plot(synthetic_noise.labels, title='Per Cluster Noise')

    def test_corrupt(self):
        synthetic = Synthetic(3, 3, number_features=2, sigma=0)
        self.assertEqual(synthetic.database.records[0].features, synthetic.database.records[1].features)
        corruption = list(np.random.normal(loc=0.0, scale=1.0, size=[9, 2]))
        synthetic.corrupt(corruption)
        self.assertNotEqual(synthetic.database.records[0].features, synthetic.database.records[1].features)

if __name__ == '__main__':
    unittest.main()