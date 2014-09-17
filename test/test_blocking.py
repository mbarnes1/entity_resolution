import unittest
import sys

sys.path.append('../../')
from entity_resolution.blocking import BlockingScheme
from entity_resolution.database import RecordDatabase

__author__ = 'mbarnes1'


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self._test_path = 'test_annotations.txt'
        self._database = RecordDatabase(self._test_path)

    def test_full_blocks(self):
        self._blocks = BlockingScheme(self._database.records)
        self.assertEqual(self._blocks._max_block_size, float('inf'))
        self.assertEqual(self._blocks.strong_blocks['poster_0000001'], {0, 3})
        self.assertEqual(self._blocks.strong_blocks['poster_0000002'], {1})
        self.assertEqual(self._blocks.strong_blocks['poster_0000003'], {2})
        self.assertEqual(self._blocks.strong_blocks['phone_8005551111'], {0})
        self.assertEqual(self._blocks.strong_blocks['phone_8005552222'], {1, 3})
        self.assertEqual(self._blocks.weak_blocks['age+1_21'], {0, 1})
        self.assertEqual(self._blocks.weak_blocks['age+1_22'], {0, 3})

    def test_clean_blocks(self):
        self._blocks = BlockingScheme(self._database.records, 1)
        self.assertRaises(KeyError, lambda: self._blocks.strong_blocks['poster_0000001'])

    def test_completeness_10000(self):
        database = RecordDatabase('test_annotations_10000.txt')
        blocks = BlockingScheme(database.records, max_block_size=200)
        used_ads = get_ads(blocks)
        self.assertEqual(used_ads, range(0, len(database.records)))


# Returns a list of all the ads in an iterable of records, in ascending order
def get_ads(blocks):
    used_ads = set()
    for _, ads in blocks.strong_blocks.iteritems():
        used_ads.update(ads)
    for _, ads in blocks.weak_blocks.iteritems():
        used_ads.update(ads)
    used_ads = list(used_ads)
    used_ads.sort()
    return used_ads

if __name__ == '__main__':
    unittest.main()