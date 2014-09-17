import unittest
import datetime
import sys

sys.path.append('../../')
from entity_resolution.utils import Record, month

__author__ = 'mbarnes1'


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self._r1 = Record()
        self._r2 = Record()
        self._annotation_full = '9552601,neworleans,Thu Jan 30 02:41:11 EST 2014,Louisiana,New Orleans,8,' \
                                '0,haley,22,60/80/100,5,6,110,DD,30,26,29,caucasian,white,blue,brunette,' \
                                'rest_type,rest_ethnicity,res_age,9802534087;5182561877,NC;NY,charlotte;' \
                                'albany,john@smith.com,www.johnsmith.com,johnsmithmedia,' \
                                'Louisiana_2014_1_30_1391067671000_6_0.jpg;Louisiana_2014_1_30_1391067671000_6_1.jpg;' \
                                'Louisiana_2014_1_30_1391067671000_6_2.jpg;Louisiana_2014_1_30_1391067671000_6_3.jpg;' \
                                'Louisiana_2014_1_30_1391067671000_6_4.jpg;Louisiana_2014_1_30_1391067671000_6_5.jpg'

    def test_initialize_from_full_annotation(self):
        self._r1.initialize_from_annotation(self._annotation_full, 0)
        self.assertSetEqual(self._r1.ads, {0})
        self.assertSetEqual(self._r1.posters, {'9552601'})
        self.assertSetEqual(self._r1.sites, {'neworleans'})
        self.assertSetEqual(self._r1.dates, {datetime.datetime(2014, 1, 30, 2, 41, 11)})
        self.assertSetEqual(self._r1.states, {'Louisiana'})
        self.assertSetEqual(self._r1.cities, {'New Orleans'})
        self.assertEqual(self._r1.perspective_1st, 0)  # currently holding at zero
        self.assertEqual(self._r1.perspective_3rd, 0)  # currently holding at zero
        self.assertSetEqual(self._r1.names, {'haley'})
        self.assertSetEqual(self._r1.ages, {22})
        self.assertSetEqual(self._r1.costs, {'60/80/100'})
        self.assertSetEqual(self._r1.heights, {66})
        self.assertSetEqual(self._r1.weights, {110})
        self.assertSetEqual(self._r1.cups, {'DD'})
        self.assertSetEqual(self._r1.chests, {30})
        self.assertSetEqual(self._r1.waists, {26})
        self.assertSetEqual(self._r1.hips, {29})
        self.assertSetEqual(self._r1.ethnicities, {'caucasian'})
        self.assertSetEqual(self._r1.skincolors, {'white'})
        self.assertSetEqual(self._r1.eyecolors, {'blue'})
        self.assertSetEqual(self._r1.haircolors, {'brunette'})
        self.assertSetEqual(self._r1.phones, {'9802534087', '5182561877'})
        self.assertSetEqual(self._r1.emails, {'john@smith.com'})
        self.assertSetEqual(self._r1.urls, {'www.johnsmith.com'})
        self.assertSetEqual(self._r1.medias, {'johnsmithmedia'})
        self.assertSetEqual(self._r1.images, {'Louisiana_2014_1_30_1391067671000_6_0.jpg',
                                              'Louisiana_2014_1_30_1391067671000_6_1.jpg',
                                              'Louisiana_2014_1_30_1391067671000_6_2.jpg',
                                              'Louisiana_2014_1_30_1391067671000_6_3.jpg',
                                              'Louisiana_2014_1_30_1391067671000_6_4.jpg',
                                              'Louisiana_2014_1_30_1391067671000_6_5.jpg'})

    def test_eq(self):
        self.assertEqual(self._r1, self._r2)
        self._r1.initialize_from_annotation(self._annotation_full, 0)
        self.assertNotEqual(self._r1, self._r2)
        self._r2.initialize_from_annotation(self._annotation_full, 0)
        self.assertEqual(self._r1, self._r2)

    def test_initialize_from_empty_annotation(self):
        annotation_empty = ',,,,,,,,,,,,,,,,,,,,,,,,,,,,,,'  # 30 fields
        self._r1.initialize_from_annotation(annotation_empty, 0)
        self._r2.add_ad(0)
        self.assertEqual(self._r1, self._r2)

    def test_initialize_from_none_annotation(self):
        annotation_none = 'none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,' \
                          'none,none,none,none,none,none,none,none,none,none,none,none,none'
        self._r1.initialize_from_annotation(annotation_none, 0)
        self._r2.add_ad(0)
        self.assertEqual(self._r1, self._r2)

    def test_merge(self):
        self._r1.merge(self._r2)
        self.assertEqual(self._r1, self._r2)
        self._r1.initialize_from_annotation(self._annotation_full, 0)
        self._r1.merge(self._r2)
        self.assertNotEqual(self._r1, self._r2)
        self._r2.merge(self._r1)
        self.assertEqual(self._r1, self._r2)

    def test_month(self):
        m = month('Apr')
        self.assertEqual(m, 4)


if __name__ == '__main__':
    unittest.main()