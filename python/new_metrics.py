from itertools import combinations
from bisect import bisect, bisect_left
__author__ = 'mbarnes1'


class NewMetrics(object):
    """
    Estimated lower bounds for pairwise precision, recall, and F1
    """
    def __init__(self, database, labels, match_function, class_balance_test):
        """
        Un/semi-supervised entity resolution metrics
        :param database: Reference to Database object
        :param match_function: Match function object
        :param class_balance_test: The predicted class balance in database
        """
        print 'Evaluating new metric...'
        self.class_balance_test = class_balance_test
        self.match_function = match_function
        self.recall_lower_bound, self.recall_lower_bound_lower_ci, self.recall_lower_bound_upper_ci = self._pairwise_recall_lower_bound()
        self.precision_lower_bound, self.precision_lower_bound_lower_ci, self.precision_lower_bound_upper_ci, self.TP_FP_match, self.TP_FP_swoosh = self._pairwise_precision_lower_bound(database, labels)
        self.f1_lower_bound = 2*self.precision_lower_bound*self.recall_lower_bound/(self.precision_lower_bound + self.recall_lower_bound)
        print 'new metric evaluated.'

    def _pairwise_recall_lower_bound(self):
        """
        Lower bounds the pairwise recall
        :return recall_lower_bound:
        """
        threshold = self.match_function.get_decision_threshold()
        print 'Lower bounding pairwise recall at threshold', threshold
        index = bisect_left(self.match_function.roc.prob, threshold)  # first occurence of this threshold
        print 'Validation set match recall', self.match_function.roc.recall[index]
        recall_lower_bound = self.match_function.roc.recall[index]
        recall_lower_bound_lower_ci = self.match_function.roc.recall_lower_ci[index]
        recall_lower_bound_upper_ci = self.match_function.roc.recall_upper_ci[index]
        return recall_lower_bound, recall_lower_bound_lower_ci, recall_lower_bound_upper_ci

    def _pairwise_precision_lower_bound(self, database, labels):
        """
        Lower bounds the pairwise precision
        :param database: Original database object
        :param labels: Predicted labels of the database, of form [record_id, cluster_id]
        :return precision_lower_bound:
        """
        cluster_to_records = dict()
        for record_id, cluster_id in labels.iteritems():
            if cluster_id in cluster_to_records:
                cluster_to_records[cluster_id].add(record_id)
            else:
                cluster_to_records[cluster_id] = {record_id}
        threshold = self.match_function.get_decision_threshold()

        print 'Lower bounding pairwise precision at threshold', threshold
        index = bisect(self.match_function.roc.prob, threshold) - 1  # first occurence of this threshold
        if index >= 0:
            match_precision_validation = self.match_function.roc.precision[index]
            match_precision_validation_upper_ci = self.match_function.roc.precision_upper_ci[index]
            match_precision_validation_lower_ci = self.match_function.roc.precision_lower_ci[index]
        elif index == -1:
            match_precision_validation = self.match_function.roc.precision[0]
            match_precision_validation_upper_ci = self.match_function.roc.precision_upper_ci[0]
            match_precision_validation_lower_ci = self.match_function.roc.precision_lower_ci[0]
        else:
            raise IndexError
        print '     Validation set match precision =', match_precision_validation
        class_balance_validation = self.match_function.roc.class_balance
        print '     Rebalancing precision for validation class balance', class_balance_validation
        print '     and test set class balance', self.class_balance_test
        match_precision_test = rebalance_precision(match_precision_validation, class_balance_validation, self.class_balance_test)
        match_precision_test_upper_ci = rebalance_precision(match_precision_validation_upper_ci, class_balance_validation, self.class_balance_test)
        match_precision_test_lower_ci = rebalance_precision(match_precision_validation_lower_ci, class_balance_validation, self.class_balance_test)
        print '     Expected test set match precision', match_precision_test
        total_swoosh_pairs = 0  # number of predicted intercluster pairs
        total_match_pairs = 0  # number of predicted intercluster pairs that directly matc
        for cluster, record_ids in cluster_to_records.iteritems():
            number_pairs = len(record_ids)*(len(record_ids)-1)/2
            print '     Cluster swoosh pairs:', number_pairs
            pairs = combinations(record_ids, 2)
            cluster_match_pairs = 0
            for counter, pair in enumerate(pairs):
                print '     Checking match ', counter, 'of', number_pairs, 'pairs to lower bound precision'
                r1 = database.records[pair[0]]
                r2 = database.records[pair[1]]
                match, _ = self.match_function.match(r1, r2)
                cluster_match_pairs += match
            print '     Cluster match pairs:', cluster_match_pairs
            total_swoosh_pairs += number_pairs
            total_match_pairs += cluster_match_pairs
        if total_swoosh_pairs:
            precision_lower_bound = match_precision_test*total_match_pairs/total_swoosh_pairs
            precision_lower_bound_upper_ci = match_precision_test_upper_ci*total_match_pairs/total_swoosh_pairs
            precision_lower_bound_lower_ci = match_precision_test_lower_ci*total_match_pairs/total_swoosh_pairs
        else:
            precision_lower_bound = 1.0
            precision_lower_bound_upper_ci = 1.0
            precision_lower_bound_lower_ci = 1.0
        print '     Total match pairs:', total_match_pairs
        print '     Total swoosh pairs:', total_swoosh_pairs
        print '     Precision lower bound:', precision_lower_bound
        return precision_lower_bound, precision_lower_bound_lower_ci, precision_lower_bound_upper_ci, \
               total_match_pairs, total_swoosh_pairs


def rebalance_recall(recall_1, class_balance_1, class_balance_2):
    """
    The expected precision and recall in dataset 2, given the precision and recall in dataset 1, and both class balances
    :param recall_1:
    :param class_balance_1: Float [0, 1.0]. P/(P+N)
    :param class_balance_2: Float [0, 1.0]. P/(P+N)
    :return recall_1:
    """
    recall_2 = recall_1  # unaffected by class balance
    return recall_2


def rebalance_precision(precision_1, class_balance_1, class_balance_2):
    """
    The expected precision and recall in dataset 2, given the precision and recall in dataset 1, and both class balances
    :param precision_1:
    :param class_balance_1: Float [0, 1.0]. P/(P+N)
    :param class_balance_2: Float [0, 1.0]. P/(P+N)
    :return precision_2:
    """
    precision_2 = (1-class_balance_1)*class_balance_2*precision_1/(class_balance_1-class_balance_1*class_balance_2-class_balance_1*precision_1+class_balance_2*precision_1)
    return precision_2


def count_pairwise_class_balance(labels):
    """
    Returns the percent of positive pairs out of all the pairs in database.
    Eventually this should be automated with density estimates, using a train database, train labels, and test database
    :param labels: Corresponding labels for the database object. Dict of [record id, label]
    :return class_balance: Percent of positive pairs in database, [0, 1.0]
    """
    print 'Calculating class balance for labels:'
    print(labels)
    number_records = len(labels)

    total_number_pairs = number_records*(number_records-1)/2
    print '     Total number of pairs:', total_number_pairs
    cluster_to_records = dict()
    for record_id, cluster_id in labels.iteritems():
        if cluster_id in cluster_to_records:
            cluster_to_records[cluster_id].add(record_id)
        else:
            cluster_to_records[cluster_id] = {record_id}
    total_number_positive_pairs = 0.0
    for _, record_ids in cluster_to_records.iteritems():
        number_cluster_pairs = len(record_ids)*(len(record_ids)-1)/2
        total_number_positive_pairs += number_cluster_pairs
    print '     Number of positive pairs:', total_number_positive_pairs
    class_balance = total_number_positive_pairs/total_number_pairs
    print '     Class balance:', class_balance
    return class_balance