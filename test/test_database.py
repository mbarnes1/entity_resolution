import unittest
import sys

sys.path.append('../../')
from entity_resolution.database import RecordDatabase, remove_indices, find_empty_entries

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
        empty_entries = find_empty_entries(lst)
        self.assertEqual([1, 3, 5], empty_entries)

    def test_size(self):
        self._database = RecordDatabase(self._test_path)
        self.assertEqual(len(self._database.records), 4)

    def test_max_size(self):
        database = RecordDatabase('test_annotations_10000.csv', 409)
        self.assertEqual(len(database.records), 409)

if __name__ == '__main__':
    unittest.main()