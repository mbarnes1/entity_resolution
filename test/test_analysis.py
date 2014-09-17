__author__ = 'mbarnes1'

import unittest
import sys
sys.path.append('../../')
from entity_resolution.pipeline import Pipeline
from entity_resolution.analysis import Analysis, get_pairs

__author__ = 'mbarnes1'


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self._test_path = 'test_annotations.txt'
        self._pipeline = Pipeline(annotations_path=self._test_path, train_size=3, match_type='strong')
        self._pipeline.run()
        self._analysis = Analysis(self._pipeline)

    def test_get_pairs(self):
        pairs = get_pairs(self._pipeline.entities)
        manual = {(0, 1), (1, 0), (0, 3), (3, 0), (1, 3), (3, 1)}
        self.assertSetEqual(pairs, manual)

    def test_pair_recall(self):
        self.assertEqual(self._analysis.pair_recall, 1)

    def test_large(self):
        pipe_large = Pipeline('test_annotations_1000.txt', train_size=100)
        pipe_large.run()
        pipe_large.analyze()
        self.assertEqual(pipe_large.analysis.pair_recall, 1)
        pipe_large.analysis.print_metrics()

    def test_plots(self):
        pipe = Pipeline(annotations_path='test_annotations_100.txt', match_type='strong', train_size=100,
                        max_block_size=100)
        pipe.run()
        pipe.analyze()
        pipe.analysis.make_plots()
        self.assertTrue(True)