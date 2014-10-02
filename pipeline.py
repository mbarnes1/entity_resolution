from database import Database
from entityresolution import EntityResolution
from metrics import Metrics
__author__ = 'mbarnes'


def main():
    regex_path = 'path_to_regex'
    train_size = 5000
    balancing = True
    max_block_size = 500
    match_type = 'weak_strong'
    db = Database(regex_path)
    training = db.sample_and_remove(train_size)
    testing = db
    er = EntityResolution()
    labels_train = fast_strong_cluster(training)
    labels_test = fast_strong_cluster(testing)
    weak_match_function = er.train(training, labels_train, train_size, balancing)
    labels_pred = er.run(testing, match_type, weak_match_function, max_block_size)
    metrics = Metrics(labels_test, labels_pred)
    weak_match_function.evaluate(testing, labels_test)


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
        indicies = record.line_indices
        strong_features = record.get_features('strong')
        if not strong_features:  # no strong features, insert singular entity
            for identifier in indicies:
                cluster_labels[identifier] = cluster_counter
            cluster_counter += 1
        for strong in strong_features:
            if strong in strong2index:
                strong2index[strong].extend(list(indicies))
            else:
                strong2index[strong] = list(indicies)
            for index in indicies:
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