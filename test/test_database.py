import unittest
import sys

sys.path.append('../../')
from entity_resolution.database import Database, remove_indices, find_in_list

__author__ = 'mbarnes1'


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self._test_path = 'test_annotations.csv'

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
        database = Database('test_annotations_10000.csv', 409)
        self.assertEqual(len(database.records), 409)

if __name__ == '__main__':
    unittest.main()