import unittest
from python.metrics import Metrics, _cluster, _intersection_size, _number_pairs, _jaccard, _linearize
from sklearn import metrics as skmetrics
import math
import numpy as np
__author__ = 'mbarnes1'


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.labels_pred = {
            0: 1,
            1: 0,
            2: 1,
            3: 1,
            4: 1,
            5: 1,
            6: 1,
            7: 0,
            8: 0,
            9: 0,
            10: 0,
            11: 2,
            12: 1,
            13: 2,
            14: 1,
            15: 2,
            16: 2}
        self.labels_true = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 1,
            7: 1,
            8: 1,
            9: 1,
            10: 1,
            11: 1,
            12: 2,
            13: 2,
            14: 2,
            15: 2,
            16: 2}
        self._n = len(self.labels_pred)
        self.metrics = Metrics(self.labels_true, self.labels_pred)

    def test_pairwise_precision_recall_f1(self):
        precision, recall, f1 = self.metrics._pairwise_precision_recall_f1()
        self.assertEqual(precision, 20.0/44)
        self.assertEqual(recall, 20.0/40)
        self.assertEqual(f1, 10.0/21)

    def test_cluster_precision_recall_f1(self):
        precision, recall, f1 = self.metrics._cluster_precision_recall_f1()
        self.assertEqual(precision, 0)
        self.assertEqual(recall, 0)
        self.assertEqual(f1, 0)

    def test_closest_cluster_precision_recall_f1(self):
        precision, recall, f1 = self.metrics._closest_cluster_precision_recall_f1()
        x = (4.0/7.0 + 5.0/9.0 + 3.0/6.0)/3.0
        self.assertEqual(precision, x)
        self.assertEqual(recall, x)
        self.assertEqual(f1, 2*x*x/(x+x))

    def test_average_author_cluster_purity(self):
        aap = 149.0/255
        acp = 193.0/340
        average_author_purity, average_cluster_purity, k = self.metrics._average_author_cluster_purity()
        self.assertEqual(average_author_purity, aap)
        self.assertEqual(average_cluster_purity, acp)
        self.assertEqual(k, (aap*acp)**0.5)

    def test_homogeneity_completeness_vmeasure(self):
        labels_true, labels_pred = _linearize(self.labels_true, self.labels_pred)
        sk_homogeneity, sk_completeness, sk_vmeasure = skmetrics.homogeneity_completeness_v_measure(labels_true,
                                                                                                    labels_pred)
        homogeneity, completeness, vmeasure = self.metrics._homogeneity_completeness_vmeasure(1)
        self.assertEqual(homogeneity, sk_homogeneity)
        self.assertEqual(completeness, sk_completeness)
        self.assertEqual(sk_vmeasure, vmeasure)

    def test_cluster(self):
        clusters = frozenset({frozenset({0, 1, 2, 3, 4, 5}), frozenset({6, 7, 8, 9, 10, 11}),
                              frozenset({12, 13, 14, 15, 16})})

        self.assertSetEqual(_cluster(self.labels_true), clusters)

    def test_intersection_size(self):
        self.assertEqual(_intersection_size(_cluster(self.labels_true), _cluster(self.labels_pred)), 20)

    def test_number_pairs(self):
        self.assertEqual(_number_pairs(_cluster(self.labels_true)), 40)

    def test_jaccard(self):
        set1 = {1, 2, 3}
        set2 = {3, 4, 5}
        set3 = {6}
        self.assertEqual(_jaccard(set1, set2), 1.0/5)
        self.assertEqual(_jaccard(set1, set3), 0)

    def test_global_merge_distance(self):
        """
        This tests GMD using the relationship to other properties specified in the original paper
        """
        fs = lambda x, y: x*y
        fm = lambda x, y: 0
        independent = frozenset({frozenset({0}), frozenset({1}), frozenset({2}), frozenset({3}), frozenset({4}),
                                 frozenset({5}), frozenset({6}), frozenset({7}), frozenset({8}), frozenset({9}),
                                 frozenset({10}), frozenset({11}), frozenset({12}), frozenset({13}), frozenset({14}),
                                 frozenset({15}), frozenset({16})})
        gmd_pairwise_precision = 1 - self.metrics.global_merge_distance(fs, fm)/self.metrics.global_merge_distance(fs, fm, S=independent)
        fs = lambda x, y: 0
        fm = lambda x, y: x*y
        gmd_pairwise_recall = 1 - self.metrics.global_merge_distance(fs, fm)/self.metrics.global_merge_distance(fs, fm, R=independent)
        pairwise_precision, pairwise_recall, f1 = self.metrics._pairwise_precision_recall_f1()
        self.assertAlmostEqual(gmd_pairwise_precision, pairwise_precision)
        self.assertAlmostEqual(gmd_pairwise_recall, pairwise_recall)

    def test_mutual_information(self):
        labels_true, labels_pred = _linearize(self.labels_true, self.labels_pred)
        mi = self.metrics._mutual_information(self.metrics._clusters_pred, self.metrics._clusters_true)
        self.assertAlmostEqual(mi, skmetrics.mutual_info_score(labels_true, labels_pred))

    def test_variation_of_information(self):
        vi = self.metrics._variation_of_information()
        h = lambda x: float(x)/self._n*math.log(float(x)/self._n)
        fs = lambda x, y: h(x+y) - h(x) - h(y)
        fm = fs
        gmd_vi = self.metrics.global_merge_distance(fs, fm)
        self.assertEqual(vi, gmd_vi)

    def test_purity(self):
        purity = self.metrics._purity()
        self.assertEqual(purity, 12.0/17)

    def test_entity_sizes(self):
        sizes_true = np.array([[5, 1],
                              [6, 2]])
        sizes_pred = np.array([[4, 1],
                               [5, 1],
                               [8, 1]])
        self.assertTrue(np.array_equal(sizes_true, self.metrics._entity_sizes_true))
        self.assertTrue(np.array_equal(sizes_pred, self.metrics._entity_sizes_pred))

    def test_plot(self):
        self.metrics.display()

if __name__ == '__main__':
    unittest.main()