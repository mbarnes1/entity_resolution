__author__ = 'mbarnes1'
import unittest
from python.pairwise_features import SurrogateMatchFunction, mean_imputation, number_matches, numerical_difference, \
    binary_match, get_x1, get_x2, get_pairwise_features, generate_pair_seed, get_precomputed_x2, levenshtein
from python.pipeline import fast_strong_cluster
from python.database import Database
from python.blocking import BlockingScheme
import numpy as np
from math import isnan


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self._database = Database('test_annotations_cleaned.csv')
        self._blocking = BlockingScheme(self._database)
        self._surrogate = SurrogateMatchFunction(0.99)

    def test_pairs(self):
        database = Database('test_annotations_10000_cleaned.csv')
        labels = fast_strong_cluster(database)
        pair_seed = generate_pair_seed(database, labels, 10000, True)
        # x1_a, x2_a, m_a = _get_pairs(database, labels, 10, balancing=True)
        # x1_b, x2_b, m_b = _get_pairs(database, labels, 10, balancing=True)
        # self.assertNotEqual(x1_a, x1_b)
        # self.assertNotEqual(x2_a, x2_b)
        # self.assertNotEqual(m_a, m_b)
        x1_a, x2_a, m_a = get_pairwise_features(database, labels, 10, balancing=True, pair_seed=pair_seed)
        x1_b, x2_b, m_b = get_pairwise_features(database, labels, 10, balancing=True, pair_seed=pair_seed)
        np.testing.assert_array_equal(x1_a, x1_b)
        np.testing.assert_array_equal(x2_a, x2_b)
        np.testing.assert_array_equal(m_a, m_b)

    def test_mean_imputation(self):
        x = np.array([[1, 2, 3, 4], [np.NaN, 4, 5, np.NaN], [1, 6, np.NaN, np.NaN]])
        m = mean_imputation(x)
        self.assertTrue((m == np.array([1, 4, 4, 4])).all())

    def test_match(self):
        r0 = self._database.records[0]
        r1 = self._database.records[1]
        r2 = self._database.records[2]
        r3 = self._database.records[3]
        labels = {
            0: 0,
            1: 0,
            2: 1,
            3: 1
        }
        self._surrogate.train(self._database, labels, 4, balancing=True)
        self.assertTrue(self._surrogate.match(r0, r3, 'strong')[0])
        self.assertTrue(self._surrogate.match(r1, r3, 'strong')[0])
        self.assertFalse(self._surrogate.match(r0, r1, 'strong')[0])
        self.assertFalse(self._surrogate.match(r0, r2, 'strong')[0])
        self.assertFalse(self._surrogate.match(r1, r2, 'strong')[0])
        self.assertFalse(self._surrogate.match(r2, r3, 'strong')[0])
        self.assertTrue(self._surrogate.match(r0, r0, 'strong')[0])
        self.assertTrue(self._surrogate.match(r1, r1, 'strong')[0])
        self.assertTrue(self._surrogate.match(r2, r2, 'strong')[0])
        self.assertTrue(self._surrogate.match(r3, r3, 'strong')[0])

    def test_test(self):
        database = Database('test_annotations_10000_cleaned.csv')
        database_train = database.sample_and_remove(5000)
        database_test = database
        labels_train = fast_strong_cluster(database_train)
        labels_test = fast_strong_cluster(database_test)
        surrogate = SurrogateMatchFunction(0.99)
        surrogate.train(database_train, labels_train, 500)
        roc = surrogate.test(database_test, labels_test, 500)
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
        self.assertEqual(x2[1], 0)
        self.assertEqual(x2[2], 1)
        self.assertEqual(x2[3], 1)
        self.assertEqual(x2[4], 1)
        self.assertEqual(x2[5], 0)
        self.assertTrue(isnan(x2[6]))
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

    def test_number_matches(self):
        x_a = {1, 2, 3}
        x_b = {3, 4, 5}
        x_c = set()
        self.assertEqual(number_matches(x_a, x_a), 3)
        self.assertEqual(number_matches(x_a, x_b), 1)
        self.assertTrue(isnan(number_matches(x_a, x_c)))

    def test_numerical_difference(self):
        x_a = {1, 2, 3}
        x_b = {4, 5, 5}
        x_c = set()
        self.assertEqual(numerical_difference(x_a, x_a), 0)
        self.assertEqual(numerical_difference(x_a, x_b), 1)
        self.assertTrue(isnan(numerical_difference(x_a, x_c)))

    def test_binary_match(self):
        x_a = {1, 2, 3}
        x_b = {3, 4, 5}
        x_c = set()
        x_d = {5}
        self.assertEqual(binary_match(x_a, x_a), 1)
        self.assertEqual(binary_match(x_a, x_b), 1)
        self.assertEqual(binary_match(x_a, x_d), 0)
        self.assertTrue(isnan(binary_match(x_a, x_c)))

    def test_levenshtein(self):
        r1 = {'Matthew'}
        r2 = {'Matt'}
        d = levenshtein(r1, r2)
        self.assertEqual(d, 3)
        r1 = {'abcd', 'efgh', 'ijkl'}
        r2 = {'abbb', 'egfe', 'i'}
        d = levenshtein(r1, r2)
        self.assertEqual(d, 2)
        d = levenshtein(r1, r1)
        self.assertEqual(d, 0)


if __name__ == '__main__':
    unittest.main()