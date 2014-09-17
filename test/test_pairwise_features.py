__author__ = 'mbarnes1'
import unittest
from pairwise_features import SurrogateMatchFunction, mean_imputation, get_x2, get_x1, exact_matches, difference_minimum
from database import RecordDatabase
from blocking import BlockingScheme
import numpy as np
from math import isnan


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self._database = RecordDatabase('test_annotations.txt')
        self._blocking = BlockingScheme(self._database.records)
        self._surrogate = SurrogateMatchFunction(self._database.records, self._blocking.strong_blocks, 6, 0.99)

    def test_mean_imputation(self):
        x = np.array([[1, 2, 3, 4], [np.NaN, 4, 5, np.NaN], [1, 6, np.NaN, np.NaN]])
        m = mean_imputation(x)
        self.assertTrue((m == np.array([1, 4, 4, 4])).all())

    def test_match(self):
        r0 = self._database.records[0]
        r1 = self._database.records[1]
        r2 = self._database.records[2]
        r3 = self._database.records[3]
        self.assertTrue(self._surrogate.match(r0, r3, 'strong'))
        self.assertTrue(self._surrogate.match(r1, r3, 'strong'))
        self.assertFalse(self._surrogate.match(r0, r1, 'strong'))
        self.assertFalse(self._surrogate.match(r0, r2, 'strong'))
        self.assertFalse(self._surrogate.match(r1, r2, 'strong'))
        self.assertFalse(self._surrogate.match(r2, r3, 'strong'))
        self.assertTrue(self._surrogate.match(r0, r0, 'strong'))
        self.assertTrue(self._surrogate.match(r1, r1, 'strong'))
        self.assertTrue(self._surrogate.match(r2, r2, 'strong'))
        self.assertTrue(self._surrogate.match(r3, r3, 'strong'))

    def test_test(self):
        database = RecordDatabase('test_annotations_1000.txt')
        blocking = BlockingScheme(database.records)
        surrogate = SurrogateMatchFunction(database.records, blocking.strong_blocks, 50, 0.99)
        roc = surrogate.test(database.records, 50)
        roc.make_plot()

    def test_get_x1(self):
        r0 = self._database.records[0]
        r1 = self._database.records[1]
        r2 = self._database.records[2]
        r3 = self._database.records[3]
        self.assertEqual(get_x1(r0, r3), True)
        self.assertEqual(get_x1(r1, r3), True)
        self.assertEqual(get_x1(r0, r1), False)
        self.assertEqual(get_x1(r0, r2), False)
        self.assertEqual(get_x1(r1, r2), False)
        self.assertEqual(get_x1(r2, r3), False)

    def test_get_x2(self):
        r0 = self._database.records[0]
        x2 = get_x2(r0, r0)
        self.assertEqual(x2[0], 1)
        # self.assertEqual(x2[1], 0)  #
        self.assertEqual(x2[2], 1)
        self.assertEqual(x2[3], 1)
        self.assertEqual(x2[4], 0)
        self.assertEqual(x2[5], 1)
        self.assertEqual(x2[6], 0)
        self.assertTrue(isnan(x2[7]))
        self.assertTrue(isnan(x2[8]))
        self.assertTrue(isnan(x2[9]))
        self.assertTrue(isnan(x2[10]))
        self.assertTrue(isnan(x2[11]))
        self.assertTrue(isnan(x2[12]))
        self.assertTrue(isnan(x2[13]))
        self.assertTrue(isnan(x2[14]))
        self.assertTrue(isnan(x2[15]))
        self.assertTrue(isnan(x2[16]))
        self.assertTrue(isnan(x2[17]))
        self.assertTrue(isnan(x2[18]))
        self.assertEqual(x2[19], 3)

    def test_exact_matches(self):
        x_a = {1, 2, 3}
        x_b = {3, 4, 5}
        x_c = set()
        self.assertEqual(exact_matches(x_a, x_a), 3)
        self.assertEqual(exact_matches(x_a, x_b), 1)
        self.assertTrue(isnan(exact_matches(x_a, x_c)))

    def test_difference_minimum(self):
        x_a = {1, 2, 3}
        x_b = {4, 5, 5}
        x_c = set()
        self.assertEqual(difference_minimum(x_a, x_a), 0)
        self.assertEqual(difference_minimum(x_a, x_b), 1)
        self.assertTrue(isnan(difference_minimum(x_a, x_c)))

if __name__ == '__main__':
    unittest.main()