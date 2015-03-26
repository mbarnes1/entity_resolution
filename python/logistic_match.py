"""
Logistic regression match function
Satisfies ICAR properties
"""
import sklearn
from sklearn import linear_model
import numpy as np
from roc import RocCurve
from pairwise_features import get_weak_pairwise_features, get_pairwise_features, generate_pair_seed
__author__ = 'mbarnes1'
if sklearn.__version__ != '0.16-git':
    raise Exception('Invalid version of sklearn')


class LogisticMatchFunction(object):
    """
    Match function based on logistic regression, with weights restricted to either the positive or negative domain to
    satisfy the ICAR properties
    """
    def __init__(self, database_train, labels_train, pair_seed, decision_threshold):
        self.roc = None
        self._logreg = linear_model.LogisticRegression(solver='lbfgs')
        self._x2_mean = self._train(database_train, labels_train, pair_seed)
        self._decision_threshold = decision_threshold

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
        :return x2_mean: Mean feature values used in imputation (i.e. to fill in missing features)
        """
        x1_train, x2_train, x2_mean = get_pairwise_features(database_train, labels_train, pair_seed)
        bounds = list()
        for _ in range(x2_train.shape[1]):
            bounds.append((None, 0))  # restrict all feature weights to be negative (convex set)
        bounds.append((None, None))  # no bounds for the constant offset term
        print 'Training Logistic Regression pairwise match function...'
        print '     Pos/Neg training sample class split: ', sum(x1_train), '/', len(x1_train) - sum(x1_train)
        print '     Enforcing weight vector bounds:'
        for counter, bound in enumerate(bounds):
            print '         Feature', counter, ': ', bounds
        print '         (last term is intercept)'
        self._logreg.bounds = bounds
        self._logreg.fit(x2_train, x1_train)
        print 'Model coefficients: ', self._logreg.coef_
        print 'Intercept: ', self._logreg.intercept_
        return x2_mean

    def test(self, database_test, labels_test, class_balance):
        """
        Get testing samples and test the surrogate match function. Evaluated with ROC curve
        :param database_test: RecordDatabase object
        :param test_size: Number of pairwise samples to use in testing
        :param class_balance: Float [0, 1.0]. Percent of matches in seed (0=all mismatch, 1=all match)
        :return RocCurve: An RocCurve object
        """
        self.class_balance_validation = class_balance
        pair_seed = generate_pair_seed(database_test, labels_test, class_balance)
        x1_test, x2_test, _ = get_pairwise_features(database_test, labels_test, pair_seed)
        x1_bar_probability = self._logreg.predict_proba(x2_test)[:, 1]
        #output = np.column_stack((x1_test, x1_bar_probability))
        #np.savetxt('roc_labels.csv', output, delimiter=",", header='label,probability', fmt='%.1i,%5.5f')
        roc = RocCurve(x1_test, x1_bar_probability)
        self.roc = roc
        sorted_indices = np.argsort(-1*x1_bar_probability)
        for sorted_index in sorted_indices:
            x1_bar = x1_bar_probability[sorted_index]
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
        np.copyto(x2, self._x2_mean, where=np.isnan(x2))  # mean imputation
        p_x1 = self._logreg.predict_proba(x2)[0, 1]
        x1_hat = p_x1 > self._decision_threshold
        if r1 == r2:
            x1_hat = True  # if records are the same, to satisfy Idempotence property
        return x1_hat, p_x1
