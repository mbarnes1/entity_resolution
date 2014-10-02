import unittest
import datetime
from record import Record, FeatureDescriptor, month
__author__ = 'mbarnes1'


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self._features_full = '9552601,neworleans,Thu Jan 30 02:41:11 EST 2014,Louisiana,New Orleans,8,' \
                              '0,haley,22,60/80/100,66,110,DD,30,26,29,caucasian,white,blue,brunette,' \
                              'rest_type,rest_ethnicity,res_age,9802534087;5182561877,NC;NY,charlotte;' \
                              'albany,john@smith.com,www.johnsmith.com,johnsmithmedia,' \
                              'Louisiana_2014_1_30_1391067671000_6_0.jpg;Louisiana_2014_1_30_1391067671000_6_1.jpg;' \
                              'Louisiana_2014_1_30_1391067671000_6_2.jpg;Louisiana_2014_1_30_1391067671000_6_3.jpg;' \
                              'Louisiana_2014_1_30_1391067671000_6_4.jpg;Louisiana_2014_1_30_1391067671000_6_5.jpg'
        self._features_full = self._features_full.split(',')
        feature_types = ('int,string,date,string,string,int,int,string,int,string,int,int,string,int,int,int,'
                               'string,string,string,string,,,,int,string,string,string,string,string,'
                               'string').split(',')
        feature_descriptor = FeatureDescriptor('', feature_types, '', '', '')
        feature_descriptor.number = len(feature_types)
        self._r0 = Record(0, feature_descriptor)
        self._r1 = Record(0, feature_descriptor)

    def test_initialize_from_full_annotation(self):
        self._r0.initialize_from_annotation(self._features_full)
        self.assertSetEqual(self._r0.line_indices, {0})
        self.assertSetEqual(self._r0.features[0], {9552601})
        self.assertSetEqual(self._r0.features[1], {'neworleans'})
        self.assertSetEqual(self._r0.features[2], {datetime.datetime(2014, 1, 30, 2, 41, 11)})
        self.assertSetEqual(self._r0.features[3], {'Louisiana'})
        self.assertSetEqual(self._r0.features[4], {'New Orleans'})
        self.assertEqual(self._r0.features[5], {8})
        self.assertEqual(self._r0.features[6], {0})
        self.assertSetEqual(self._r0.features[7], {'haley'})
        self.assertSetEqual(self._r0.features[8], {22})
        self.assertSetEqual(self._r0.features[9], {'60/80/100'})
        self.assertSetEqual(self._r0.features[10], {66})
        self.assertSetEqual(self._r0.features[11], {110})
        self.assertSetEqual(self._r0.features[12], {'DD'})
        self.assertSetEqual(self._r0.features[13], {30})
        self.assertSetEqual(self._r0.features[14], {26})
        self.assertSetEqual(self._r0.features[15], {29})
        self.assertSetEqual(self._r0.features[16], {'caucasian'})
        self.assertSetEqual(self._r0.features[17], {'white'})
        self.assertSetEqual(self._r0.features[18], {'blue'})
        self.assertSetEqual(self._r0.features[19], {'brunette'})
        self.assertSetEqual(self._r0.features[23], {9802534087, 5182561877})
        self.assertSetEqual(self._r0.features[26], {'john@smith.com'})
        self.assertSetEqual(self._r0.features[27], {'www.johnsmith.com'})
        self.assertSetEqual(self._r0.features[28], {'johnsmithmedia'})
        self.assertSetEqual(self._r0.features[29], {'Louisiana_2014_1_30_1391067671000_6_0.jpg',
                                                    'Louisiana_2014_1_30_1391067671000_6_1.jpg',
                                                    'Louisiana_2014_1_30_1391067671000_6_2.jpg',
                                                    'Louisiana_2014_1_30_1391067671000_6_3.jpg',
                                                    'Louisiana_2014_1_30_1391067671000_6_4.jpg',
                                                    'Louisiana_2014_1_30_1391067671000_6_5.jpg'})

    def test_eq(self):
        self.assertEqual(self._r0, self._r1)
        self._r0.initialize_from_annotation(self._features_full)
        self.assertNotEqual(self._r0, self._r1)
        self._r1.initialize_from_annotation(self._features_full)
        self.assertEqual(self._r0, self._r1)

    def test_initialize_from_empty_annotation(self):
        annotation_empty = ',,,,,,,,,,,,,,,,,,,,,,,,,,,,,'.split(',')
        self._r0.initialize_from_annotation(annotation_empty)
        self.assertEqual(self._r0, self._r1)

    def test_merge(self):
        self._r0.merge(self._r1)
        self.assertEqual(self._r0, self._r1)
        self._r0.initialize_from_annotation(self._features_full)
        self._r0.merge(self._r1)
        self.assertNotEqual(self._r0, self._r1)
        self._r1.merge(self._r0)
        self.assertEqual(self._r0, self._r1)

    def test_month(self):
        m = month('Apr')
        self.assertEqual(m, 4)


if __name__ == '__main__':
    unittest.main()