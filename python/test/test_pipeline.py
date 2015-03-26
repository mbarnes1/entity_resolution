import unittest
from entityresolution import EntityResolution
from database import Database
from blocking import BlockingScheme
__author__ = 'mbarnes1'


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self._test_path = 'test_annotations_cleaned.csv'
        self._database = Database(self._test_path)
        self._blocking = BlockingScheme(self._database)
        self._er = EntityResolution()


