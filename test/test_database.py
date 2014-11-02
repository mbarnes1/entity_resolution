import unittest
import sys

from database import Synthetic, Database, remove_indices, find_in_list
import os

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
        synthetic = Synthetic(10, 10, sigma=0)  # 100 records, 10 entities, 10 records each
        self.assertEqual(len(synthetic.truth), 100)
        self.assertEqual(synthetic.truth[0], 0)
        self.assertEqual(synthetic.truth[99], 9)
        self.assertEqual(len(synthetic.database.records), 100)
        self.assertEqual(synthetic.database.records[0].features, synthetic.database.records[1].features)  # line indices won't (and shouldn't) match. Different records
        self.assertEqual(synthetic.database.records[98].features, synthetic.database.records[98].features)
        self.assertNotEqual(synthetic.database.records[9].features, synthetic.database.records[10].features)
        self.assertEqual(synthetic.database.feature_descriptor.number, 10)

    def test_dump(self):
        database = Database(self._test_path)
        database.dump('test_annotations_dump.csv')
        database2 = Database('test_annotations_dump.csv')
        os.remove('test_annotations_dump.csv')
        self.assertEqual(database, database2)


if __name__ == '__main__':
    unittest.main()