"""
All pairwise features and functions. This includes training and testing the match function.
"""
from sklearn import linear_model
import numpy as np
from numpy.random import choice
import random
from roc import RocCurve
from itertools import izip
__author__ = 'mbarnes1'


class SurrogateMatchFunction(object):
    """
    The all important match function, named after the surrogate labels it uses for training
    """
    def __init__(self, decision_threshold=1):
        self.x2_mean = []
        self.roc = []
        self.logreg = linear_model.LogisticRegression()
        self.decision_threshold = decision_threshold

    def train(self, database_train, labels_train, number_samples, balancing=True, pair_seed=None):
        """
        Get training samples and trains the surrogate match function
        :param database_train: Training database
        :param labels_train: Dictionary of [identier, cluster label]
        :param number_samples: Number of samples to use in training
        :param balancing: Boolean, whether to balance match/mismatch classes
        :param pair_seed: List of pairs to use. Otherwise random
        :return pairs: Set of pairs. Each pair is a list with two entries, the two ad identifiers
        """
        x1_train, x2_train, self.x2_mean = get_pairwise_features(database_train, labels_train, number_samples,
                                                                 balancing, pair_seed=pair_seed)
        self.logreg.fit(x2_train, x1_train)

    def test(self, database_test, labels_test, test_size=1000):
        """
        Get testing samples and test the surrogate match function. Evaluated with ROC curve
        :param database_test: RecordDatabase object
        :param test_size: Number of pairwise samples to use in testing
        :return RocCurve: An RocCurve object
        """
        x1_test, x2_test, _ = get_pairwise_features(database_test, labels_test, test_size, True)
        x1_bar_probability = self.logreg.predict_proba(x2_test)[:, 1]
        #output = np.column_stack((x1_test, x1_bar_probability))
        #np.savetxt('roc_labels.csv', output, delimiter=",", header='label,probability', fmt='%.1i,%5.5f')
        return RocCurve(x1_test, x1_bar_probability)

    def match(self, r1, r2, match_type):
        """
        Determines if two records match
        :param r1: Record object
        :param r2: Record object
        :param match_type: Match type use in ER algorithm. String 'strong', 'weak', or 'weak_strong'
        :return: False or True, whether r1 and r2 match
        :return p_x1: Probability of weak match
        :return strength: Type of matches that occurred. 'strong', 'weak', 'both', or 'none'
        """
        x1 = get_x1(r1, r2)
        if np.isnan(x1):
            x1 = False
        x2 = get_x2(r1, r2)
        np.copyto(x2, self.x2_mean, where=np.isnan(x2))  # mean imputation
        p_x1 = self.logreg.predict_proba(x2)[0, 1]
        x1_hat = p_x1 > self.decision_threshold
        if x1 and x1_hat:
            strength = 'both'  # both match
            return True, p_x1, strength
        elif x1:
            strength = 'strong'
            if match_type == 'strong':
                return True, p_x1, strength
            else:
                return False, p_x1, strength
        elif x1_hat:
            strength = 'weak'
            if match_type == 'weak':
                return True, p_x1, strength
            else:
                return False, p_x1, strength
        else:
            strength = 'none'
            return False, p_x1, strength


def generate_pair_seed(database, labels_train, number_samples, balancing):
    """
    Generates list of pairs to seed from. Useful for removing random process noise for tests on related datasets
    :param database: Database object
    :param labels_train: Cluster labels of database. Dictionary [identifier, cluster label]
    :param number_samples: Number of samples in pair seed
    :param balancing: Boolean, whether to balance match/mismatch classes
    :return pair_seed: List of pairs, where each pair is a vector of form (identifierA, identifierB)
    """
    if len(database.records)*(len(database.records)-1)/2 < number_samples:
        raise Exception('Number of requested pairs exceeds number of pairs available in database')
    print 'Generating pairwise seed of length', number_samples
    pairs = set()
    cluster_to_records = dict()
    line_indices = database.records.keys()
    print 'Creating cluster to identifier hash table'
    for index, cluster in labels_train.iteritems():
        if cluster in cluster_to_records:
            cluster_to_records[cluster].append(index)
        else:
            cluster_to_records[cluster] = [index]
    cluster_keys = cluster_to_records.keys()
    cluster_sizes = []  # probability of randomly selecting a pair from a cluster
    number_pairs = 0
    print 'Calculating number of pairs in each cluster, for cluster sampling probability'
    for key in cluster_keys:
        records = cluster_to_records[key]
        cluster_pairs = float(len(records))*(len(records)-1)/2
        cluster_sizes.append(cluster_pairs)
        number_pairs += cluster_pairs
    cluster_prob = [size/number_pairs for size in cluster_sizes]
    counter = 0
    while len(pairs) < number_samples:
        print 'Sampling pair', counter
        to_balance = random.choice([True, False])
        if balancing and to_balance:
            print 'Sampling pair from within cluster'
            cluster = choice(cluster_keys, p=cluster_prob)
            pair = None
            while not pair:
                pair = tuple(choice(cluster_to_records[cluster], size=2, replace=False))
                pair_flipped = (pair[1], pair[0])
                if (pair[0] != pair[1]) and \
                        (pair not in pairs) and \
                        (pair_flipped not in pairs):  # valid pair, not used before
                    pairs.add(pair)
        else:
            print 'Sampling pair from between clusters'
            pair = None
            while not pair:
                pair = tuple(choice(line_indices, size=2, replace=False))
                pair_flipped = (pair[1], pair[0])
                if (labels_train[pair[0]] != labels_train[pair[1]]) and \
                        (pair[0] != pair[1]) and \
                        (pair not in pairs) and \
                        (pair_flipped not in pairs):  # not in cluster, valid pair, not used before
                    pairs.add(pair)
        counter += 1
    pairs = list(pairs)
    return pairs


def get_pairwise_features(database, labels_train, number_samples, balancing, pair_seed=None):
    """
    Randomly samples pairs of records, without replacement.
    Can balance classes, using labels_train
    :param database: Database object to sample from
    :param labels_train: Cluster labels of the dictionary form [identifier, cluster label]
    :param number_samples: Number of samples to return
    :param balancing: Boolean, whether to balance match/mismatch classes
    :param pair_seed: (Optional) pairs to use. Otherwise generates randomly
    :return x1: Vector of strong features, values takes either 0 or 1
    :return x2: n x m Matrix of weak features, where n is number_samples and m is number of weak features
    :return m: Mean imputation vector of weak features, 1 x number of weak features
    """
    if pair_seed is None:  # randomly draw new pairs
        if len(database.records)*(len(database.records)-1)/2 < number_samples:
            raise Exception('Number of requested pairs exceeds number of pairs available in database')
        pairs = generate_pair_seed(database, labels_train, number_samples, balancing)
    else:  # use pair seed
        if number_samples > len(pair_seed):
            raise Exception('Number requested samples larger than seed')
        pairs = pair_seed[:number_samples]
    x1 = list()
    x2 = list()
    for pair in pairs:
        r1 = database.records[pair[0]]
        r2 = database.records[pair[1]]
        c1 = labels_train[next(iter(r1.line_indices))]  # cluster r1 belongs to
        c2 = labels_train[next(iter(r2.line_indices))]  # cluster r2 belongs to
        _x1 = int(c1 == c2)
        x1.append(_x1)
        x2.append(get_x2(r1, r2))
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    m = mean_imputation(x2)
    return x1, x2, m


def get_x2(r1, r2):
    """
    Calculates the weak feature vector x2 based on differences between two records
    :param r1: Record object
    :param r2: Record object
    :return x2: 1-D vector with m entries, where m is number of weak features.
    """
    x2 = list()
    for index, (f1, f2, pairwise_use, strength) in enumerate(izip(r1.features, r2.features,
                                                                  r1.feature_descriptor.pairwise_uses,
                                                                  r1.feature_descriptor.strengths)):
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


def get_x1(r1, r2):
    """
    Calculates the strong feature value x1 based on differences between two records
    If any strong feature matches, x1 is True
    :param r1: Record object
    :param r2: Record object
    :return x1: 1 (match), 0 (mismatch), or NaN (at least one record has no strong features, not enough info)
    """
    x1 = list()
    for f1, f2, pairwise_use, strength in izip(r1.features, r2.features,
                                               r1.feature_descriptor.pairwise_uses,
                                               r1.feature_descriptor.strengths):
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