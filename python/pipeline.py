from database import Database, SyntheticDatabase
from entityresolution import EntityResolution, fast_strong_cluster, weak_connected_components
from logistic_match import LogisticMatchFunction
from decision_tree_match import TreeMatchFunction
from random_forest_match import ForestMatchFunction
from blocking import BlockingScheme
from pairwise_features import generate_pair_seed
from metrics import Metrics
from new_metrics import NewMetrics, count_pairwise_class_balance
from copy import deepcopy
import numpy as np
import cProfile
import matplotlib.pyplot as plt
__author__ = 'mbarnes'


def main():
    """
    Runs a single entity resolution on data (real or synthetic) using a match function (logistic regression, decision
    tree, or random forest)
    """
    data_type = 'real'
    decision_threshold = 0.7
    train_class_balance = 0.5
    max_block_size = 1000
    cores = 2
    if data_type == 'synthetic':
        database_train = SyntheticDatabase(100, 10, 10)
        corruption = 0.1
        corruption_array = corruption*np.random.normal(loc=0.0, scale=1.0, size=[1000,
                                                       database_train.database.feature_descriptor.number])
        database_train.corrupt(corruption_array)

        database_validation = SyntheticDatabase(100, 10, 10)
        corruption_array = corruption*np.random.normal(loc=0.0, scale=1.0, size=[1000,
                                                       database_validation.database.feature_descriptor.number])
        database_validation.corrupt(corruption_array)

        database_test = SyntheticDatabase(10, 10, 10)
        corruption_array = corruption*np.random.normal(loc=0.0, scale=1.0, size=[1000,
                                                       database_test.database.feature_descriptor.number])
        database_test.corrupt(corruption_array)
        labels_train = database_train.labels
        labels_validation = database_validation.labels
        labels_test = database_test.labels
        database_train = database_train.database
        database_validation = database_validation.database
        database_test = database_test.database
        single_block = True
    elif data_type == 'real':
        # Uncomment to use all features (annotations and LM)
        #database_train = Database('../data/trafficking/cluster_subsample0_10000.csv', header_path='../data/trafficking/cluster_subsample_header_all.csv')
        #database_validation = Database('../data/trafficking/cluster_subsample1_10000.csv', header_path='../data/trafficking/cluster_subsample_header_all.csv')
        #database_test = Database('../data/trafficking/cluster_subsample1_10000.csv', header_path='../data/trafficking/cluster_subsample_header_all.csv')

        # Uncomment to only use annotation features
        #database_train = Database('../data/trafficking/cluster_subsample0_10000.csv', header_path='../data/trafficking/cluster_subsample_header_annotations.csv')
        #database_validation = Database('../data/trafficking/cluster_subsample1_10000.csv', header_path='../data/trafficking/cluster_subsample_header_annotations.csv')
        #database_test = Database('../data/trafficking/cluster_subsample2_10000.csv', header_path='../data/trafficking/cluster_subsample_header_annotations.csv')

        # Uncomment to only use LM features
        database_train = Database('../data/trafficking/cluster_subsample0_10000.csv', header_path='../data/trafficking/cluster_subsample_header_LM.csv')
        database_validation = Database('../data/trafficking/cluster_subsample1_10000.csv', header_path='../data/trafficking/cluster_subsample_header_LM.csv')
        database_test = Database('../data/trafficking/cluster_subsample1_10000.csv', header_path='../data/trafficking/cluster_subsample_header_LM.csv')

        labels_train = fast_strong_cluster(database_train)
        labels_validation = fast_strong_cluster(database_validation)
        labels_test = fast_strong_cluster(database_test)
        single_block = False
    else:
        Exception('Invalid experiment type'+data_type)

    entities = deepcopy(database_test)
    blocking_scheme = BlockingScheme(entities, max_block_size, single_block=single_block)

    train_seed = generate_pair_seed(database_train, labels_train, train_class_balance, require_direct_match=True, max_minor_class=5000)
    validation_seed = generate_pair_seed(database_validation, labels_validation, 0.5, require_direct_match=True, max_minor_class=5000)
    # forest_all = ForestMatchFunction(database_all_train, labels_train, train_seed, decision_threshold)
    # forest_all.test(database_all_validation, labels_validation, validation_seed)
    # tree_all = TreeMatchFunction(database_all_train, labels_train, train_seed, decision_threshold)
    # tree_all.test(database_all_validation, labels_validation, validation_seed)
    # logistic_all = LogisticMatchFunction(database_all_train, labels_train, train_seed, decision_threshold)
    # logistic_all.test(database_all_validation, labels_validation, validation_seed)

    forest_annotations = ForestMatchFunction(database_train, labels_train, train_seed, decision_threshold)
    roc = forest_annotations.test(database_validation, labels_validation, validation_seed)
    #roc.make_plot()
    #plt.show()

    # tree_annotations = TreeMatchFunction(database_annotations_train, labels_train, train_seed, decision_threshold)
    # tree_annotations.test(database_annotations_validation, labels_validation, validation_seed)
    # logistic_annotations = LogisticMatchFunction(database_annotations_train, labels_train, train_seed, decision_threshold)
    # logistic_annotations.test(database_annotations_validation, labels_validation, validation_seed)

    # forest_LM = ForestMatchFunction(database_LM_train, labels_train, train_seed, decision_threshold)
    # forest_LM.test(database_LM_validation, labels_validation, validation_seed)
    # tree_LM = TreeMatchFunction(database_LM_train, labels_train, train_seed, decision_threshold)
    # tree_LM.test(database_LM_validation, labels_validation, validation_seed)
    # logistic_LM = LogisticMatchFunction(database_LM_train, labels_train, train_seed, decision_threshold)
    # logistic_LM.test(database_LM_validation, labels_validation, validation_seed)

    # forest_all.roc.write_rates('match_forest_all.csv')
    # tree_all.roc.write_rates('match_tree_all.csv')
    # logistic_all.roc.write_rates('match_logistic_all.csv')
    #
    # forest_annotations.roc.write_rates('match_forest_annotations.csv')
    # tree_annotations.roc.write_rates('match_tree_annotations.csv')
    # logistic_annotations.roc.write_rates('match_logistic_annotations.csv')
    #
    # forest_LM.roc.write_rates('match_forest_LM.csv')
    # tree_LM.roc.write_rates('match_tree_LM.csv')
    # logistic_LM.roc.write_rates('match_logistic_LM.csv')
    # ax = forest_all.roc.make_plot()
    # _ = tree_all.roc.make_plot(ax=ax)
    # _ = logistic_all.roc.make_plot(ax=ax)
    # plt.show()
    #forest_annotations.roc.make_plot()
    #plt.show()

    #entities.merge(strong_labels)

    #er = EntityResolution()
    #weak_labels = er.run(entities, match_function, blocking_scheme, cores=cores)
    weak_labels = weak_connected_components(database_test, forest_annotations, blocking_scheme)
    entities.merge(weak_labels)
    #strong_labels = fast_strong_cluster(entities)
    #entities.merge(strong_labels)

    # out = open('ER.csv', 'w')
    # out.write('phone,cluster_id\n')
    # for cluster_counter, (entity_id, entity) in enumerate(entities.records.iteritems()):
    #     phone_index = 21
    #     for phone in entity.features[phone_index]:
    #         out.write(str(phone)+','+str(cluster_counter)+'\n')
    # out.close()

    print 'Metrics using strong features as surrogate label. Entity resolution run using weak and strong features'
    metrics = Metrics(labels_test, weak_labels)
    # estimated_test_class_balance = count_pairwise_class_balance(labels_test)
    # new_metrics = NewMetrics(database_all_test, weak_labels, forest_all, estimated_test_class_balance)
    metrics.display()
    # new_metrics.display()


if __name__ == '__main__':
    cProfile.run('main()')