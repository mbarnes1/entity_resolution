from database import Database
from entityresolution import EntityResolution, fast_strong_cluster
from logistic_match import LogisticMatchFunction
from pairwise_features import generate_pair_seed
from metrics import Metrics
from new_metrics import NewMetrics, count_pairwise_class_balance
from copy import deepcopy
__author__ = 'mbarnes'


def main():
    regex_path = 'test/test_annotations_10000.csv'
    train_size = 6000
    validation_size = 2000
    decision_threshold = 0.9
    train_class_balance = 0.5
    max_block_size = 10000
    cores = 2

    database = Database(regex_path)
    database_train = database.sample_and_remove(train_size)
    database_validation = database.sample_and_remove(validation_size)
    database_test = database

    labels_train = fast_strong_cluster(database_train)
    labels_validation = fast_strong_cluster(database_validation)


    train_seed = generate_pair_seed(database_train, labels_train, train_class_balance)
    match_function = LogisticMatchFunction(database_train, labels_train, train_seed, decision_threshold)
    match_roc = match_function.test(database_validation, labels_validation, 0.5)


    strong_labels = fast_strong_cluster(database_test)
    database_test.merge(strong_labels)

    er = EntityResolution()
    weak_labels = er.run(deepcopy(database_test), match_function, max_block_size=max_block_size, cores=cores)
    database_test.merge(weak_labels)

    print 'Metrics using strong features as surrogate label. Entity resolution run using weak and strong features'
    metrics = Metrics(strong_labels, weak_labels)
    _print_metrics(metrics)
    estimated_test_class_balance = count_pairwise_class_balance(strong_labels)
    new_metrics = NewMetrics(database_test, match_function, estimated_test_class_balance)


def _print_metrics(metrics):
    """
    Prints metrics to console
    :param metrics: Metrics object
    """
    print 'Pairwise precision:', metrics.pairwise_precision
    print 'Pairwise recall:', metrics.pairwise_recall
    print 'Pairwise F1:', metrics.pairwise_f1, '\n'
    print 'Cluster precision:', metrics.cluster_precision
    print 'Cluster recall:', metrics.cluster_recall
    print 'Cluster F1:', metrics.cluster_f1, '\n'
    print 'Closest cluster preicision:', metrics.closest_cluster_precision
    print 'Closest cluster recall:', metrics.closest_cluster_recall
    print 'Closest cluster F1:', metrics.closest_cluster_f1, '\n'
    print 'Average cluster purity:', metrics.acp
    print 'Average author purity:', metrics.aap
    print 'K:', metrics.k, '\n'
    print 'Homogeneity:', metrics.homogeneity
    print 'Completeness:', metrics.completeness
    print 'V-Measure:', metrics.vmeasure, '\n'
    print 'Variation of Information:', metrics.variation_of_information, '\n'
    print 'Purity:', metrics.purity


if __name__ == '__main__':
    main()