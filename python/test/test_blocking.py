import unittest
from blocking import BlockingScheme
from database import Database
import numpy as np
__author__ = 'mbarnes1'


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self._test_path = 'test_annotations_cleaned.csv'
        self._database = Database(self._test_path)

    def test_full_blocks(self):
        self._blocks = BlockingScheme(self._database)
        self.assertEqual(self._blocks._max_block_size, np.Inf)
        self.assertEqual(self._blocks.strong_blocks['PosterID_1'], {0, 3})
        self.assertEqual(self._blocks.strong_blocks['PosterID_2'], {1})
        self.assertEqual(self._blocks.strong_blocks['PosterID_3'], {2})
        self.assertEqual(self._blocks.strong_blocks['PhoneNumber_8005551111'], {0})
        self.assertEqual(self._blocks.strong_blocks['PhoneNumber_8005552222'], {1, 3})

    def test_clean_blocks(self):
        self._blocks = BlockingScheme(self._database, 1)
        self.assertRaises(KeyError, lambda: self._blocks.strong_blocks['poster_0000001'])

    def test_completeness_10000(self):
        database = Database('test_annotations_10000_cleaned.csv')
        blocks = BlockingScheme(database, max_block_size=200)
        used_ads = get_records(blocks)
        self.assertEqual(used_ads, range(0, len(database.records)))

    def test_single_block(self):
        blocks = BlockingScheme(self._database, single_block=True)
        self.assertEqual(len(blocks.strong_blocks), 0)
        self.assertEqual(len(blocks.weak_blocks), 1)
        self.assertEqual(blocks.weak_blocks['All'], {0, 1, 2, 3})


def get_records(blocks):
    """
    Returns a list of all the ads in an iterable of records, in ascending order
    :param blocks: Block object
    :return used_records: List of records that were blocked
    """
    used_records = set()
    for _, ads in blocks.strong_blocks.iteritems():
        used_records.update(ads)
    for _, ads in blocks.weak_blocks.iteritems():
        used_records.update(ads)
    used_records = list(used_records)
    used_records.sort()
    return used_records

if __name__ == '__main__':
    unittest.main()