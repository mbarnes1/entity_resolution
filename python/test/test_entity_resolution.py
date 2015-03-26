import unittest
import sys
print sys.path
from entityresolution import EntityResolution, merge_duped_records
from database import Database
from pairwise_features import generate_pair_seed
from blocking import BlockingScheme
from pipeline import fast_strong_cluster
from logistic_match import LogisticMatchFunction
__author__ = 'mbarnes1'
from copy import deepcopy


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self._test_path = 'test_annotations_cleaned.csv'
        self._database = Database(self._test_path)
        self._labels = fast_strong_cluster(self._database)
        self._blocking = BlockingScheme(self._database)
        self._er = EntityResolution()
        decision_threshold = 1.0
        pair_seed = generate_pair_seed(self._database, self._labels, 0.5)
        self._match_function = LogisticMatchFunction(self._database, self._labels, pair_seed, decision_threshold)

    def test_run(self):
        strong_clusters = fast_strong_cluster(self._database)
        database_copy = deepcopy(self._database)
        database_copy.merge(strong_clusters)
        self._er.run(database_copy, self._match_function, cores=2)
        r0 = self._database.records[0]
        r1 = self._database.records[1]
        r2 = self._database.records[2]
        r3 = self._database.records[3]
        r0.merge(r1)
        r0.merge(r3)
        manual = {r0, r2}
        self.assertTrue(test_object_set(manual, set(self._er.entities)))

    def test_rswoosh(self):
        strong_clusters = fast_strong_cluster(self._database)
        database_copy = deepcopy(self._database)
        database_copy.merge(strong_clusters)
        records = set()
        for _, record in database_copy.records.iteritems():
            records.add(record)
        self._er._match_function = self._match_function
        swooshed = self._er.rswoosh(records)
        # Compare to manually merged records
        r0 = self._database.records[0]
        r1 = self._database.records[1]
        r2 = self._database.records[2]
        r3 = self._database.records[3]
        r1.merge(r3)
        r0.merge(r1)
        merged = {r0, r2}
        self.assertEqual(len(swooshed), len(merged))
        self.assertTrue(test_object_set(merged, swooshed))

    def test_merge_duped_records(self):
        """
        Merges all entities containing the same record identifier
        """
        strong_clusters = fast_strong_cluster(self._database)
        database_copy = deepcopy(self._database)
        database_copy.merge(strong_clusters)
        self._er._match_function = self._match_function
        records = set()
        for _, record in database_copy.records.iteritems():
            records.add(record)
        swooshed = self._er.rswoosh(records)
        # Compare to manually constructed clusters with duplicates
        r0 = self._database.records[0]
        r1 = self._database.records[1]
        r2 = self._database.records[2]
        r3 = self._database.records[3]
        r0.merge(r1)
        r1.merge(r3)
        premerge = {0: r0,
                    1: r1,
                    2: r2,
                    3: r3}
        merged = merge_duped_records(premerge)
        self.assertEqual(len(merged), len(swooshed))
        self.assertTrue(test_object_set(merged, swooshed))

    def test_deep_copy(self):
        records_copy = deepcopy(self._database.records)
        r1 = records_copy[0]
        self.assertEqual(r1, self._database.records[0])
        r1.features[0].add('Santa Clause')
        self.assertNotEqual(r1, self._database.records[0])

    def test_completeness(self):
        database = Database('test_annotations_10000_cleaned.csv', max_records=1000)
        database_train = database.sample_and_remove(800)
        database_test = database
        labels_train = fast_strong_cluster(database_train)
        labels_test = fast_strong_cluster(database_test)
        er = EntityResolution()
        pair_seed = generate_pair_seed(database_train, labels_train, 0.5)
        match_function = LogisticMatchFunction(database_train, labels_train, pair_seed, 0.99)
        labels_pred = er.run(database_test, match_function, cores=2)
        number_fast_strong_records = len(labels_train) + len(labels_test)
        self.assertEqual(number_fast_strong_records, 1000)
        self.assertEqual(sorted((labels_train.keys() + labels_test.keys())), range(0, 1000))
        number_swoosh_records = len(get_ids(er.entities))
        self.assertEqual(number_swoosh_records, len(database_test.records))
        self.assertEqual(get_ids(er.entities), sorted(labels_test.keys()))
        self.assertEqual(get_ids(er.entities), sorted(labels_pred.keys()))

    def test_fast_strong_cluster(self):
        labels_pred = fast_strong_cluster(self._database)
        labels_true = {
            0: 0,
            1: 0,
            2: 1,
            3: 0
        }
        self.assertEqual(labels_pred, labels_true)

    def test_fast_strong_cluster_large(self):
        database = Database('test_annotations_10000_cleaned.csv', max_records=1000)
        database_train = database.sample_and_remove(800)
        database_test = database
        labels_train = fast_strong_cluster(database_train)
        labels_test = fast_strong_cluster(database_test)
        self.assertEqual(len(labels_train), len(database_train.records))
        self.assertEqual(len(labels_test), len(database_test.records))


def get_ids(records):
    """
    Returns a sorted list of all the ids in an iterable of records, in ascending order
    :param records: Iterable of record objects
    :return ids: List of all original record identifiers
    """
    ids = []
    for r in records:
        for identifier in r.line_indices:
            ids.append(identifier)
    ids.sort()
    return ids


def test_object_set(set1, set2_destructive):
    """
    Tests whether two sets of record objects are equal. Used for debugging
    :param set1: set of record objects
    :param set2_destructive: set of record objects, which would be modified w/o deepcopy operation
    """
    set2 = deepcopy(set2_destructive)
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
