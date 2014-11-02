import unittest
from pipeline import fast_strong_cluster
from entityresolution import EntityResolution
from database import Database
from blocking import BlockingScheme
from metrics import _cluster
__author__ = 'mbarnes1'


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self._test_path = 'test_annotations.csv'
        self._database = Database(self._test_path)
        self._blocking = BlockingScheme(self._database)
        self._er = EntityResolution()

    def test_fast_strong_cluster(self):
        labels_pred = fast_strong_cluster(self._database)
        labels_true = {
            0: 0,
            1: 0,
            2: 1,
            3: 0
        }
        self.assertEqual(labels_pred, labels_true)

    def test_fast_strong_cluster_large(self):
        database = Database('test_annotations_10000.csv', max_records=1000)
        database_train = database.sample_and_remove(500)
        database_test = database
        labels_train = fast_strong_cluster(database_train)
        labels_test = fast_strong_cluster(database_test)
        er = EntityResolution()
        match_function = er.train(database_train, labels_train, 1000, True)
        labels_pred = er.run(database_test, match_function, 0.99, match_type='strong', cores=2)
        self.assertEqual(_cluster(labels_pred), _cluster(labels_test))
