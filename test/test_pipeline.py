import unittest
import sys

sys.path.append('../../')
from entity_resolution.pipeline import Pipeline, fast_strong_cluster, merge_duped_records
from entity_resolution.database import RecordDatabase
from entity_resolution.blocking import BlockingScheme
__author__ = 'mbarnes1'
from copy import deepcopy


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self._test_path = 'test_annotations.txt'
        self._database = RecordDatabase(self._test_path)
        self._blocking = BlockingScheme(self._database.records)
        self._pipeline = Pipeline(self._test_path, train_size=6)
        self._pipeline._init_large_files()

    def test_run(self):
        self._pipeline.run(2)
        r0 = self._pipeline.database.records[0]
        r1 = self._pipeline.database.records[1]
        r2 = self._pipeline.database.records[2]
        r3 = self._pipeline.database.records[3]
        r0.merge(r1)
        r0.merge(r3)
        manual = {r0, r2}
        self.assertTrue(test_object_set(manual, self._pipeline.entities))

    def test_rswoosh(self):
        r0 = self._pipeline.database.records[0]
        r1 = self._pipeline.database.records[1]
        r2 = self._pipeline.database.records[2]
        r3 = self._pipeline.database.records[3]
        records = {r0, r1, r2, r3}
        swooshed = self._pipeline.rswoosh(records)
        r1.merge(r3)
        r0.merge(r1)
        merged = {r0, r2}
        self.assertEqual(len(swooshed), len(merged))
        self.assertTrue(test_object_set(merged, swooshed))

    def test_merge_duped_records(self):
        r0 = self._pipeline.database.records[0]
        r1 = self._pipeline.database.records[1]
        r2 = self._pipeline.database.records[2]
        r3 = self._pipeline.database.records[3]
        records = {r0, r1, r2, r3}
        swooshed = self._pipeline.rswoosh(records)
        r0.merge(r1)
        r1.merge(r3)
        r3.merge(r0)
        premerge = {0: r0,
                    1: r1,
                    2: r2,
                    3: r3}
        merged = merge_duped_records(premerge)
        self.assertEqual(len(merged), len(swooshed))
        self.assertTrue(test_object_set(merged, swooshed))

    def test_fast_strong_merge(self):
        merged = fast_strong_cluster(self._pipeline.database.records)
        r0 = self._pipeline.database.records[0]
        r1 = self._pipeline.database.records[1]
        r2 = self._pipeline.database.records[2]
        r3 = self._pipeline.database.records[3]
        r0.merge(r1)
        r0.merge(r3)
        manual = {r0, r2}
        self.assertTrue(test_object_set(merged, manual))

    def test_deep_copy(self):
        records_copy = deepcopy(self._pipeline.database.records)
        r1 = records_copy[0]
        self.assertEqual(r1, self._pipeline.database.records[0])
        r1.add_name('Santa Clause')
        self.assertNotEqual(r1, self._pipeline.database.records[0])

    def test_completeness(self):
        pipe = Pipeline(annotations_path='test_annotations_1000.txt', match_type='strong')
        pipe.run(2)
        pipe.analyze()
        number_fast_strong_records = sum(pipe.analysis.strong_cluster_sizes[:, 0] * pipe.analysis.strong_cluster_sizes[:, 1])
        self.assertEqual(number_fast_strong_records, pipe.analysis.number_records)
        self.assertEqual(get_ads(pipe.strong_clusters), range(0, pipe.analysis.number_records))
        number_swoosh_records = sum(pipe.analysis.entity_sizes[:, 0] * pipe.analysis.entity_sizes[:, 1])
        self.assertEqual(number_swoosh_records, pipe.analysis.number_records)
        self.assertEqual(get_ads(pipe.entities), range(0, pipe.analysis.number_records))
        pipe.analysis.make_plots()
        self.assertTrue(test_object_set(pipe.strong_clusters, pipe.entities))


# Returns a list of all the ads in an iterable of records, in ascending order
def get_ads(records):
    ads = []
    for r in records:
        for ad in r.ads:
            ads.append(ad)
    ads.sort()
    return ads


def test_object_set(input1, input2):
    set1 = deepcopy(input1)
    set2 = deepcopy(input2)
    if len(set1) != len(set2):
        return False
    for obj1 in set1:
        for obj2 in set2:
            if obj1 == obj2:
                set2.discard(obj2)
                break
        else:  # obj1 not in set2
            return False
    return True

if __name__ == '__main__':
    unittest.main()