import unittest
import sys
print sys.path
from entityresolution import EntityResolution, merge_duped_records
from database import Database
from blocking import BlockingScheme
from pipeline import fast_strong_cluster
__author__ = 'mbarnes1'
from copy import deepcopy


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self._test_path = 'test_annotations_cleaned.csv'
        self._database = Database(self._test_path)
        self._labels = fast_strong_cluster(self._database)
        self._blocking = BlockingScheme(self._database)
        self._er = EntityResolution()
        self._er._match_type = 'strong'
        self._match_function = self._er.train(self._database, self._labels, 4, class_balance=0.5)

    def test_run(self):
        self._er.run(self._database, self._match_function, 0.99, 'strong', cores=2)
        r0 = self._database.records[0]
        r1 = self._database.records[1]
        r2 = self._database.records[2]
        r3 = self._database.records[3]
        r0.merge(r1)
        r0.merge(r3)
        manual = {r0, r2}
        self.assertTrue(test_object_set(manual, set(self._er.entities)))

    def test_rswoosh(self):
        r0 = self._database.records[0]
        r1 = self._database.records[1]
        r2 = self._database.records[2]
        r3 = self._database.records[3]
        records = {r0, r1, r2, r3}
        (swooshed, _, _) = self._er.rswoosh(deepcopy(records))
        r1.merge(r3)
        r0.merge(r1)
        merged = {r0, r2}
        self.assertEqual(len(swooshed), len(merged))
        self.assertTrue(test_object_set(merged, swooshed))
        (swooshed_random, _, _) = self._er.rswoosh(records, guarantee_random=True)
        self.assertTrue(test_object_set(swooshed, swooshed_random))

    def test_merge_duped_records(self):
        r0 = self._database.records[0]
        r1 = self._database.records[1]
        r2 = self._database.records[2]
        r3 = self._database.records[3]
        records = {r0, r1, r2, r3}
        (swooshed, _, _) = self._er.rswoosh(records)
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
        match_function = er.train(database_train, labels_train, 4, class_balance=0.5)
        labels_pred = er.run(database_test, match_function, 0.99, 'strong', cores=2)
        number_fast_strong_records = len(labels_train) + len(labels_test)
        self.assertEqual(number_fast_strong_records, 1000)
        self.assertEqual(sorted((labels_train.keys() + labels_test.keys())), range(0, 1000))
        number_swoosh_records = len(get_ids(er.entities))
        self.assertEqual(number_swoosh_records, len(database_test.records))
        self.assertEqual(get_ids(er.entities), sorted(labels_test.keys()))

    # def test_decision_plot(self):
    #     database = Database('test_annotations_10000_cleaned.csv', max_records=1000)
    #     db_train = database.sample_and_remove(500)
    #     db_test = database
    #     labels_train = fast_strong_cluster(db_train)
    #     er = EntityResolution()
    #     weak_match_function = er.train(db_train, labels_train, 1000, True)
    #     er.run(db_test, weak_match_function, 0.99, 'strong', max_block_size=50)
    #     er.plot_decisions()


# Returns a sorted list of all the ids in an iterable of records, in ascending order
def get_ids(records):
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
