from database import Database
from entityresolution import EntityResolution
from metrics import Metrics
__author__ = 'mbarnes'



def main():
    regex_path = 'test/test_annotations_10000.csv'
    train_size = 500
    test_size = 500
    balancing = True
    max_block_size = 100
    match_type = 'weak_strong'
    decision_threshold = 0.9
    db = Database(regex_path, max_records=1000)
    db_training = db.sample_and_remove(train_size)
    db_testing = db
    er = EntityResolution()
    labels_train = fast_strong_cluster(db_training)
    labels_test = fast_strong_cluster(db_testing)
    weak_match_function = er.train(db_training, labels_train, train_size, balancing)
    labels_pred = er.run(db_testing, weak_match_function, decision_threshold, match_type, max_block_size)
    metrics = Metrics(labels_test, labels_pred)
    _print_metrics(metrics)
    #metrics.display()
    roc = weak_match_function.test(db_testing, labels_test, test_size)
    #roc.make_plot()


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


def fast_strong_cluster(database):
    """
    Merges records with any strong features in common. Used as surrogate ground truth in CHT application
    Equivalent to running Swoosh using only strong matches, except much faster because of explicit graph exploration
    :param database: Database object
    :return labels: Cluster labels in the form of a dictionary [identifier, cluster_label]
    """
    strong2index = dict()  # [swoosh index, set of ad indices]  node --> edges
    index2strong = dict()  # [ad index, list of swoosh indices]  edge --> nodes (note an edge can lead to multiple nodes)
    cluster_counter = 0
    cluster_labels = dict()
    for _, record in database.records.iteritems():
        indices = record.line_indices
        strong_features = record.get_features('strong')
        if not strong_features:  # no strong features, insert singular entity
            for identifier in indices:
                cluster_labels[identifier] = cluster_counter
            cluster_counter += 1
        for strong in strong_features:
            if strong in strong2index:
                strong2index[strong].extend(list(indices))
            else:
                strong2index[strong] = list(indices)
            for index in indices:
                if index in index2strong:
                    index2strong[index].append(strong)
                else:
                    index2strong[index] = [strong]

    # Determine connected components
    explored_strong = set()  # swooshed records already explored
    explored_indices = set()
    to_explore = list()  # ads to explore
    for index, strong_features in index2strong.iteritems():
        if index not in explored_indices:
            explored_indices.add(index)
            cc = set()
            cc.add(index)
            to_explore.extend(list(strong_features))
            while to_explore:
                strong = to_explore.pop()
                if strong not in explored_strong:
                    explored_strong.add(strong)
                    connected_indices = strong2index[strong]
                    for connected_index in connected_indices:
                        if connected_index not in explored_indices:
                            cc.add(connected_index)
                            explored_indices.add(connected_index)
                            to_explore.extend(list(index2strong[connected_index]))

            # Found complete connected component, save labels
            for c in cc:
                record = database.records[c]
                for identifier in record.line_indices:
                    cluster_labels[identifier] = cluster_counter
            cluster_counter += 1
    return cluster_labels

if __name__ == '__main__':
    main()