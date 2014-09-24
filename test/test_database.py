import unittest
import sys

sys.path.append('../../')
from entity_resolution.database import RecordDatabase, remove_indices, find_in_list

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

    # def test_get_strong_features(self):
    #     database = RecordDatabase(self._test_path)
    #     strong_features = database.get_strong_features(database.records[0])
    #     self.assertEqual(strong_features, {'PosterID_1', 'PhoneNumber_8005551111'})
    #
    # def test_get_weak_features(self):
    #     database = RecordDatabase(self._test_path)
    #     weak_features = database.get_weak_features(database.records[0])
    #     self.assertEqual(weak_features, {'site_neworleans', 'Date_2014-01-30 02:41:11', 'State_Louisiana',
    #                                      'City_New Orleans', 'Name_haley', 'Age_22', 'Images_img1', 'Images_img2',
    #                                      'Images_img3'})

    def test_size(self):
        database = RecordDatabase(self._test_path)
        self.assertEqual(len(database.records), 4)

    def test_max_size(self):
        database = RecordDatabase('test_annotations_10000.csv', 409)
        self.assertEqual(len(database.records), 409)

if __name__ == '__main__':
    unittest.main()