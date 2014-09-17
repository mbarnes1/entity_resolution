import unittest
import sys

sys.path.append('../../')
from entity_resolution.database import RecordDatabase

__author__ = 'mbarnes1'


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self._test_path = 'test_annotations.txt'
        self._database = RecordDatabase(self._test_path)

    def test_size(self):
        self.assertEqual(len(self._database.records), 4)

    def test_max_size(self):
        database = RecordDatabase('test_annotations_10000.txt', 409)
        self.assertEqual(len(database.records), 409)

if __name__ == '__main__':
    unittest.main()