"""
All pairwise features and functions. This includes training and testing the match function.
"""
from sklearn import linear_model
import numpy as np
import random
import warnings
from roc import RocCurve
from copy import deepcopy
from itertools import izip
__author__ = 'mbarnes1'


class SurrogateMatchFunction(object):
    """
    The all important match function, named after the surrogate labels it uses for training
    """
    def __init__(self, database, strong_blocks, train_size, decision_threshold):
        self._remaining_strong_blocks = deepcopy(strong_blocks)
        self._database = database
        self._used_pairs = set()  # pairs used, so as not to repeat samples in training and testing
        self._x1_train = []
        self._x2_train = []
        self.x2_mean = []
        self.roc = []
        self.logreg = linear_model.LogisticRegression()
        self._train(train_size)
        self.decision_threshold = decision_threshold

    def _train(self, train_size):
        """
        Get training samples and trains the surrogate match function
        :param train_size: Number of pairwise samples to use in training
        """
        self._x1_train, self._x2_train, self.x2_mean = self._get_pairs(train_size, True)
        self.logreg.fit(self._x2_train, self._x1_train)

    def test(self, test_size):
        """
        Get testing samples and test the surrogate match function. Evaluated with ROC curve
        :param database: RecordDatabase object
        :param test_size: Number of pairwise samples to use in testing
        :return RocCurve: An RocCurve object
        """
        x1_test, x2_test, _ = self._get_pairs(test_size, True)
        x1_bar_probability = self.logreg.predict_proba(x2_test)[:, 1]
        #output = np.column_stack((x1_test, x1_bar_probability))
        #np.savetxt('roc_labels.csv', output, delimiter=",", header='label,probability', fmt='%.1i,%5.5f')
        return RocCurve(x1_test, x1_bar_probability)

    def _get_pairs(self, number_samples, balancing):
        """
        Pseudo-randomly samples pairs of records, without replacement.
        Can balance classes, approximated by a strong feature match.
        :param number_samples: Int, number of samples to return
        :param balancing: Boolean, whether to balance classes or not
        :return x1: Vector of strong features, values takes either 0 or 1
        :return x2: n x m Matrix of weak features, where n is number_samples and m is number of weak features
        :return m: Mean imputation vector of weak features, 1 x number of weak features
        """
        number_records = len(self._database.records)
        x1 = list()
        x2 = list()
        while len(x1) < number_samples:
            to_balance = random.choice([True, False])
            if balancing and to_balance:
                while self._remaining_strong_blocks:
                    _, block = self._remaining_strong_blocks.popitem()  # Deep copied in init, safe to destructively pop
                    if len(block) >= 2:
                        p1 = block.pop()
                        p2 = block.pop()
                        pair = (p1, p2)
                        break
                else:
                    balancing = False
                    warnings.warn('Not enough strong blocks for full balancing')
                    pair = (random.randint(0, number_records-1), random.randint(0, number_records-1))
            else:
                pair = (random.randint(0, number_records-1), random.randint(0, number_records-1))
            if pair not in self._used_pairs and pair[0] != pair[1]:  # if pair is valid
                self._used_pairs.add(pair)
                pair_flipped = (pair[1], pair[0])
                self._used_pairs.add(pair_flipped)
                r1 = self._database.records[pair[0]]
                r2 = self._database.records[pair[1]]
                _x1 = self.get_x1(r1, r2)
                if _x1 != np.NaN:  # are there strong features in both ads?
                    x1.append(_x1)
                    x2.append(self.get_x2(r1, r2))
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        m = mean_imputation(x2)
        return x1, x2, m

    def match(self, r1, r2, match_type):
        """
        Determines if two records match
        :param r1: Record object
        :param r2: Record object
        :param match_type: Match type use in ER algorithm. String 'strong', 'weak', or 'weak_strong'
        :return: False or True, whether r1 and r2 match
        """
        x1 = self.get_x1(r1, r2)
        if np.isnan(x1):
            x1 = False
        if match_type == 'strong':  # If only using strong matches, this is easy
            if x1:
                return True
            else:
                return False
        x2 = self.get_x2(r1, r2)
        np.copyto(x2, self.x2_mean, where=np.isnan(x2))  # mean imputation
        p_x1 = self.logreg.predict_proba(x2)[0, 1]
        x1_hat = p_x1 > self.decision_threshold
        if match_type == 'weak':
            if x1_hat:
                return True
            else:
                return False
        else:
            if x1 or x1_hat:
                return True
            else:
                return False

    def get_x2(self, r1, r2):
        """
        Calculates the weak feature vector x2 based on differences between two records
        :param r1: Record object
        :param r2: Record object
        :return x2: 1-D vector with m entries, where m is number of weak features.
        """
        x2 = list()
        for index, (f1, f2, pairwise_use, strength) in enumerate(izip(r1.features, r2.features,
                                                                      self._database.pairwise_uses,
                                                                      self._database.feature_strengths)):
            if strength == 'weak':
                if pairwise_use == 'binary_match':
                    x2.append(binary_match(f1, f2))
                elif pairwise_use == 'numerical_difference':
                    x2.append(numerical_difference(f1, f2))
                elif pairwise_use == 'number_matches':
                    x2.append(number_matches(f1, f2))
                elif pairwise_use == 'special_date_difference':
                    x2.append(date_difference(f1, f2))
                else:
                    raise Exception('Invalid pairwise use: ' + pairwise_use)
        x2 = np.asarray(x2)
        return x2

    def get_x1(self, r1, r2):
        """
        Calculates the strong feature value x1 based on differences between two records
        If any strong feature matches, x1 is True
        :param r1: Record object
        :param r2: Record object
        :return x1: 1 (match), 0 (mismatch), or NaN (at least one record has no strong features, not enough info)
        """
        x1 = list()
        for f1, f2, pairwise_use, strength in izip(r1.features, r2.features, self._database.pairwise_uses,
                                                   self._database.feature_strengths):
            if strength == 'strong':
                if (pairwise_use == 'binary_match') | (pairwise_use == 'number_matches'):
                    x1.append(binary_match(f1, f2))
                else:
                    raise Exception('Invalid pairwise use for strong features: ' + pairwise_use)
        x1 = np.asarray(x1)
        if np.isnan(x1).all():  # if all nan, return nan
            x1 = np.nan
        else:
            x1 = bool(np.nansum(x1))  # else return true/false
        return x1


def mean_imputation(x):
    """
    Determines the column means of all non-NaN entries in matrix
    :param x: n x m Matrix
    :return m: 1-D vector with m entries, the column means of non-NaN entries in x
    """
    m = np.nansum(x, axis=0)/np.sum(~np.isnan(x), axis=0)
    m = np.nan_to_num(m)  # if all one feature was NaN, mean will also be NaN. Replace with 0
    # Fill in NaN values. No longer necessary for numpy >= v1.9
    np.copyto(x, m, where=np.isnan(x))
    return m


def binary_match(features_1, features_2):
    """
    Returns True if any feature matches between two feature sets.
    Satisfies ICAR properties
    :param features_1: Set of features
    :param features_2: Set of features
    :return: True, False, or NaN (not enough info to make decision, if either set is empty)
    """
    if bool(features_1) & bool(features_2):
        x = bool(features_1 & features_2)
    else:
        x = np.nan
    return x


def number_matches(feat1, feat2):
    """
    Intersection size of two feature sets
    Satisfies ICAR properties
    :param feat1: Set of features
    :param feat2: Set of features
    :return: Int or NaN (not enough info to make decision, if either set is empty)
    """
    if bool(feat1) & bool(feat2):
        x = len(feat1 & feat2)  # intersection
    else:
        x = np.nan
    return x


def numerical_difference(feat1, feat2):
    """
    Minimum pairwise distance between two numerical feature sets
    Satisfies ICAR properties
    :param feat1: Set of features
    :param feat2: Set of features
    :return: Float or NaN (not enough info to make decision, if either set is empty)
    """
    if bool(feat1) & bool(feat2):
        x = np.Inf
        for f1 in feat1:
            for f2 in feat2:
                if abs(f1 - f2) < x:
                    x = abs(f1 - f2)
    else:
        x = np.nan
    return x


def date_difference(feat1, feat2):
    """
    Minimum pairwise disteance between two date feature sets, in seconds
    Satisfies ICAR properties
    :param feat1: Set of date features
    :param feat2: Set of date features
    :return: Float or NaN (not enough info to make decision, if either set is empty)
    """
    if bool(feat1) & bool(feat2):
        x = np.Inf
        for date1 in feat1:  # recall dates in a set, don't need to use iteritemes()
            for date2 in feat2:
                x = min(x, abs((date1-date2).seconds))
    else:
        x = np.nan
    return x