"""
Decision tree match function
Satisfies ICAR properties
"""
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from roc import RocCurve
from pairwise_features import get_weak_pairwise_features, get_pairwise_features, generate_pair_seed
__author__ = 'mbarnes1'


class ForestMatchFunction(object):
    """
    Match function based on logistic regression, with weights restricted to either the positive or negative domain to
    satisfy the ICAR properties
    """
    def __init__(self, database_train, labels_train, pair_seed, decision_threshold):
        self.roc = None
        self._classifier = RandomForestClassifier()
        self._x_mean = self._train(database_train, labels_train, pair_seed)
        self._decision_threshold = decision_threshold
        self.ICAR = False  # does not satisfy ICAR properties

    def get_recall(self):
        """
        Returns the recall and 95% bounds at the current threshold
        :return recall_lower_bound:
        :return recall_lower_bound_lower_ci: Recall 95% lower bound
        :return recall_lower_bound_upper_ci: Recall 95% upper bound
        """
        recall_lower_bound, recall_lower_bound_lower_ci, recall_lower_bound_upper_ci = \
            self.roc.get_recall(self._decision_threshold)
        return recall_lower_bound, recall_lower_bound_lower_ci, recall_lower_bound_upper_ci

    def get_precision(self):
        """
        Returns the precision and 95% bounds at the current threshold
        :return precision_lower_bound:
        :return precision_lower_bound_lower_ci: Precision 95% lower bound
        :return precision_lower_bound_upper_ci: Precision 95% upper bound
        """
        match_precision_validation, match_precision_validation_lower_ci, match_precision_validation_upper_ci = \
            self.roc.get_precision(self._decision_threshold)
        return match_precision_validation, match_precision_validation_lower_ci, match_precision_validation_upper_ci

    def set_decision_threshold(self, decision_threshold):
        """
        Modifies the match decision threshold after initialization
        :param decision_threshold: New decision threshold
        """
        self._decision_threshold = decision_threshold

    def get_decision_threshold(self):
        """
        Returns the decision threshold
        :return self._decision_threshold:
        """
        return self._decision_threshold

    def _train(self, database_train, labels_train, pair_seed):
        """
        Gets training samples and trains the surrogate match function
        :param database_train: Training database
        :param labels_train: Dictionary of [identier, cluster label]
        :param pair_seed: List of pairs to use
        :return x_mean: Mean feature values used in imputation (i.e. to fill in missing features)
        """
        y, x, x_mean = get_pairwise_features(database_train, labels_train, pair_seed)
        print 'Training decision tree pairwise match function...'
        print '     Pos/Neg training sample class split: ', sum(y), '/', len(y) - sum(y)
        self._classifier.fit(x, y)
        print 'Match function training complete.'
        return x_mean

    def test(self, database_test, labels_test, pair_seed):
        """
        Get testing samples and test the surrogate match function. Evaluated with ROC curve
        :param database_test: RecordDatabase object
        :param labels_test: Corresponding labels, of dict form [record_id, label]
        :param pair_seed: List of pairs, where each pair is a tuple of form (identifierA, identifierB)
        :return RocCurve: An RocCurve object
        """
        y_test, x_test, _ = get_pairwise_features(database_test, labels_test, pair_seed)
        self.class_balance_validation = float(sum(y_test))/len(y_test)
        x_bar_probability = self._classifier.predict_proba(x_test)[:, 1]
        #output = np.column_stack((x1_test, x1_bar_probability))
        #np.savetxt('roc_labels.csv', output, delimiter=",", header='label,probability', fmt='%.1i,%5.5f')
        roc = RocCurve(y_test, x_bar_probability)
        self.roc = roc
        sorted_indices = np.argsort(-1*x_bar_probability)
        for sorted_index in sorted_indices:
            x1_bar = x_bar_probability[sorted_index]
            pair = pair_seed[sorted_index]
            print 'Test pair P(match) = ', x1_bar
            database_test.records[pair[0]].display(indent='     ')
            print '     ----'
            database_test.records[pair[1]].display(indent='     ')
        return roc

    def match(self, r1, r2):
        """
        Determines if two records match
        :param r1: Record object
        :param r2: Record object
        :return: False or True, whether r1 and r2 match
        :return p_x1: Probability of weak match
        """
        # x1 = get_x1(r1, r2)
        # if np.isnan(x1):
        #     x1 = False
        x2 = get_weak_pairwise_features(r1, r2)
        np.copyto(x2, self._x_mean, where=np.isnan(x2))  # mean imputation
        p_x1 = self._classifier.predict_proba(x2)[0, 1]
        x1_hat = p_x1 > self._decision_threshold
        if r1 == r2:
            x1_hat = True  # if records are the same, to satisfy Idempotence property
        return x1_hat, p_x1
