from database import Database, SyntheticDatabase
from entityresolution import EntityResolution, fast_strong_cluster, weak_connected_components
from logistic_match import LogisticMatchFunction
from decision_tree_match import TreeMatchFunction
from blocking import BlockingScheme
from pairwise_features import generate_pair_seed
from metrics import Metrics
from new_metrics import NewMetrics, count_pairwise_class_balance
from copy import deepcopy
import numpy as np
__author__ = 'mbarnes'


def main():
    type = 'real'  # synthetic or real
    train_size = 500
    validation_size = 250
    decision_threshold = 0.999
    train_class_balance = 0.5
    max_block_size = 1000
    cores = 2

    if type == 'synthetic':
        database = SyntheticDatabase(100, 10, 10)
        corruption = 0.1
        corruption_array = corruption*np.random.normal(loc=0.0, scale=1.0, size=[1000,
                                                       database.database.feature_descriptor.number])
        database.corrupt(corruption_array)
    else:
        regex_path = 'test/test_annotations_10000_cleaned.csv'
        #database = Database(regex_path, max_records=1000)
        database_train = Database('../data/trafficking/cluster_subsample0_10000.csv', header_path='../data/trafficking/cluster_subsample0_10000_header.csv')
        database_validation = Database('../data/trafficking/cluster_subsample1_10000.csv', header_path='../data/trafficking/cluster_subsample1_10000_header.csv')
        database_test = Database('../data/trafficking/cluster_subsample1_10000.csv', header_path='../data/trafficking/cluster_subsample1_10000_header.csv')

    if type == 'synthetic':
        labels_train = database_train.labels
        labels_validation = database_validation.labels
        labels_test = database_test.labels
        database_train = database_train.database
        database_validation = database_validation.database
        database_test = database_test.database
    else:
        labels_train = fast_strong_cluster(database_train)
        labels_validation = fast_strong_cluster(database_validation)
        labels_test = fast_strong_cluster(database_test)

    entities = deepcopy(database_test)
    blocking_scheme = BlockingScheme(entities, max_block_size, single_block=False)

    train_seed = generate_pair_seed(database_train, labels_train, train_class_balance, require_direct_match=True)
    match_function = TreeMatchFunction(database_train, labels_train, train_seed, decision_threshold)
    match_function.test(database_validation, labels_validation, 0.5)
    match_function.roc.make_plot()


    #entities.merge(strong_labels)

    #er = EntityResolution()
    #weak_labels = er.run(entities, match_function, blocking_scheme, cores=cores)
    weak_labels = weak_connected_components(entities, match_function, blocking_scheme)
    entities.merge(weak_labels)

    print 'Metrics using strong features as surrogate label. Entity resolution run using weak and strong features'
    metrics = Metrics(labels_test, weak_labels)
    estimated_test_class_balance = count_pairwise_class_balance(labels_test)
    new_metrics = NewMetrics(database_test, weak_labels, match_function, estimated_test_class_balance)
    metrics.display()
    new_metrics.display()


if __name__ == '__main__':
    main()