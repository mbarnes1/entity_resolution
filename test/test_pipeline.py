import unittest
from pipeline import fast_strong_cluster
from entityresolution import EntityResolution
from database import Database
from blocking import BlockingScheme
__author__ = 'mbarnes1'


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self._test_path = 'test_annotations.csv'
        self._database = Database(self._test_path)
        self._blocking = BlockingScheme(self._database)
        self._er = EntityResolution(self._test_path)

    def test_fast_strong_cluster(self):
        labels_pred = fast_strong_cluster(self._database)
        labels_true = {
            0: 0,
            1: 0,
            2: 1,
            3: 0
        }
        self.assertEqual(labels_pred, labels_true)
