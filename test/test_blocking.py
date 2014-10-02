import unittest
from blocking import BlockingScheme
from database import Database

__author__ = 'mbarnes1'


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self._test_path = 'test_annotations.csv'
        self._database = Database(self._test_path)

    def test_full_blocks(self):
        self._blocks = BlockingScheme(self._database)
        self.assertEqual(self._blocks._max_block_size, float('inf'))
        self.assertEqual(self._blocks.strong_blocks['PosterID_1'], {0, 3})
        self.assertEqual(self._blocks.strong_blocks['PosterID_2'], {1})
        self.assertEqual(self._blocks.strong_blocks['PosterID_3'], {2})
        self.assertEqual(self._blocks.strong_blocks['PhoneNumber_8005551111'], {0})
        self.assertEqual(self._blocks.strong_blocks['PhoneNumber_8005552222'], {1, 3})

    def test_clean_blocks(self):
        self._blocks = BlockingScheme(self._database, 1)
        self.assertRaises(KeyError, lambda: self._blocks.strong_blocks['poster_0000001'])

    def test_completeness_10000(self):
        database = Database('test_annotations_10000.csv')
        blocks = BlockingScheme(database, max_block_size=200)
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