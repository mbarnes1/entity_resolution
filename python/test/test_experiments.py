import unittest
from experiments import SyntheticExperiment, get_pairwise_class_balance
from database import SyntheticDatabase
__author__ = 'mbarnes1'


class MyTestCase(unittest.TestCase):
    def setUp(self):
        # Create synthetic database, run ER on it
        self.experiment = SyntheticExperiment(10, 10, 5)  # a entities, b records per entity, 5 thresholds

    def test_get_pairwise_class_balance(self):
        synthetic_database = SyntheticDatabase(10, 10, number_features=2, sigma=0)
        class_balance = get_pairwise_class_balance(synthetic_database.database, synthetic_database.labels)
        self.assertEqual(class_balance, 9.0/99.0)

if __name__ == '__main__':
    unittest.main()