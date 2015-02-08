"""
All pairwise features and functions. This includes training and testing the match function.
"""
import sys
print sys.path
import sklearn
from sklearn import linear_model
import numpy as np
from math import e
from roc import RocCurve
from itertools import izip
import Levenshtein
__author__ = 'mbarnes1'
if sklearn.__version__ != '0.16-git':
    raise Exception('Invalid version of sklearn')


class SurrogateMatchFunction(object):
    """
    The all important match function, named after the surrogate labels it uses for training
    """
    def __init__(self, decision_threshold=1):
        self.x2_mean = []
        self.roc = []
        self.logreg = linear_model.LogisticRegression(solver='lbfgs')
        self.decision_threshold = decision_threshold

    def train(self, database_train, labels_train, pair_seed):
        """
        Get training samples and trains the surrogate match function
        :param database_train: Training database
        :param labels_train: Dictionary of [identier, cluster label]
        :param pair_seed: List of pairs to use
        :return pairs: Set of pairs. Each pair is a list with two entries, the two ad identifiers
        """
        x1_train, x2_train, self.x2_mean = get_pairwise_features(database_train, labels_train, pair_seed)
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
        self.logreg.bounds = bounds
        self.logreg.fit(x2_train, x1_train)
        print 'Model coefficients: ', self.logreg.coef_
        print 'Intercept: ', self.logreg.intercept_

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
        x1_bar_probability = self.logreg.predict_proba(x2_test)[:, 1]
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

    def match(self, r1, r2, match_type):
        """
        Determines if two records match
        :param r1: Record object
        :param r2: Record object
        :param match_type: Match type use in ER algorithm. String 'exact', 'strong', 'weak', or 'weak_strong'
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
        if r1 == r2:
            strength = 'exact'  # if records are the same, to satisfy Idempotence property
            return True, p_x1, strength
        elif x1 and x1_hat:
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


def generate_pair_seed(database, labels, class_balance):
    """
    Generates list of pairs to seed from. Useful for removing random process noise for tests on related datasets
    :param database: Database object
    :param labels: Cluster labels of database. Dictionary [identifier, cluster label]
    :param class_balance: Float [0, 1.0]. Percent of matches in seed (0=all mismatch, 1=all match)
    :return pair_seed: List of pairs, where each pair is a vector of form (identifierA, identifierB)
    """
    print 'Generating pairwise seed with class balance', class_balance
    pairs = set()
    cluster_to_records = dict()
    line_indices = database.records.keys()
    print '     Creating cluster to identifier hash table'
    for index, cluster in labels.iteritems():
        if cluster in cluster_to_records:
            cluster_to_records[cluster].append(index)
        else:
            cluster_to_records[cluster] = [index]
    cluster_keys = cluster_to_records.keys()
    cluster_pair_sizes = []  # number of remaining pairs in each cluster
    print '     Calculating number of pairs in each cluster, for cluster sampling probability'
    for key in cluster_keys:
        records = cluster_to_records[key]
        cluster_pairs = float(len(records))*(len(records)-1)/2
        cluster_pair_sizes.append(cluster_pairs)
    total_number_pairs = len(database.records)*(len(database.records)-1)/2
    number_remaining_pos_pairs = sum(cluster_pair_sizes)
    total_number_neg_pairs = total_number_pairs - number_remaining_pos_pairs
    print '     Number of available intracluster (pos) pairs:', number_remaining_pos_pairs
    print '     Number of available intercluster (neg) pairs:', total_number_neg_pairs
    print '     Actual class balance,', float(number_remaining_pos_pairs)/total_number_pairs
    if class_balance is None:  # use the full database (or at least up to a limit), balanced as is
        print '     No class balancing. Using actual class balance.'
        class_balance = float(number_remaining_pos_pairs)/total_number_pairs
    if float(number_remaining_pos_pairs)/total_number_pairs < class_balance:  # limited by pos class (most common)
        number_requested_pos_class = number_remaining_pos_pairs  # no more than 10000
        number_requested_neg_class = number_requested_pos_class*(1-class_balance)/class_balance
    else:
        number_requested_neg_class = total_number_neg_pairs  # no more than 10000
        number_requested_pos_class = class_balance*number_requested_neg_class/(1-class_balance)
    if min(number_requested_neg_class, number_requested_pos_class) > 1000:
        number_requested_neg_class /= min(number_requested_neg_class, number_requested_pos_class)/1000
        number_requested_pos_class /= min(number_requested_neg_class, number_requested_pos_class)/1000
    number_requested_pos_class = int(number_requested_pos_class)
    number_requested_neg_class = int(number_requested_neg_class)
    print '     Number of requested intracluster (pos) pairs:', number_requested_pos_class
    print '     Number of requested intercluster (neg) pairs:', number_requested_neg_class

    cluster_prob = [size/number_remaining_pos_pairs for size in cluster_pair_sizes]
    cluster_prob = np.array(cluster_prob)
    within_cluster_pairs = 0
    between_cluster_pairs = 0
    print '     Choosing indices to balance.'
    number_requested_samples = number_requested_neg_class+number_requested_pos_class
    to_balance_indices = np.random.choice(range(0, number_requested_samples), number_requested_pos_class, replace=False)  # indices of seed to balance
    samples_to_balance = np.zeros(number_requested_samples)
    samples_to_balance[to_balance_indices] = 1
    for balance_this_sample in samples_to_balance:
        if balance_this_sample:
            print '     Trying to sample pair from within cluster...',
            cluster_index = np.random.choice(range(0, len(cluster_keys)), p=cluster_prob)
            cluster = cluster_keys[cluster_index]
            pair = None
            while not pair:
                pair = tuple(np.random.choice(cluster_to_records[cluster], size=2, replace=False))
                pair_flipped = (pair[1], pair[0])
                if (pair[0] != pair[1]) and \
                        (pair not in pairs) and \
                        (pair_flipped not in pairs):  # valid pair, not used before
                    print 'successful.'
                    pairs.add(pair)
                    within_cluster_pairs += 1
                    print '         Updating this cluster probability from', cluster_prob[cluster_index],
                    cluster_prob[cluster_index] -= 1.0/number_remaining_pos_pairs
                    if abs(cluster_prob[cluster_index]) < 0.0000001:  # if within E-7 of zero
                        cluster_prob[cluster_index] = 0
                    print 'to', cluster_prob[cluster_index]
                    number_remaining_pos_pairs -= 1
                    normalizer = sum(cluster_prob)
                    cluster_prob = cluster_prob/normalizer
                else:
                    pair = None
        else:
            pair = None
            while not pair:
                print '     Trying to sample pair from between clusters...',
                pair = tuple(np.random.choice(line_indices, size=2, replace=False))
                pair_flipped = (pair[1], pair[0])
                if (labels[pair[0]] != labels[pair[1]]) and \
                        (pair[0] != pair[1]) and \
                        (pair not in pairs) and \
                        (pair_flipped not in pairs):  # not in cluster, valid pair, not used before
                    pairs.add(pair)
                    between_cluster_pairs += 1
                    print 'successful.'
                else:
                    pair = None
        database.records[pair[0]].display(indent='              ')
        print '               ----'
        database.records[pair[1]].display(indent='              ')
        print '         Number of pairs within:', within_cluster_pairs, '   between:', between_cluster_pairs
    pairs = list(pairs)
    print 'Finished generating pair seed.'
    print '     Total number of within cluster (pos) pairs: ', within_cluster_pairs
    print '     Total number of between cluster (neg) pairs: ', between_cluster_pairs
    return pairs


def get_pairwise_features(database, labels_train, pair_seed):
    """
    Randomly samples pairs of records, without replacement.
    Can balance classes, using labels_train
    :param database: Database object to sample from
    :param labels_train: Cluster labels of the dictionary form [identifier, cluster label]
    :param number_samples: Number of samples to return
    :param pair_seed: Pairs to use.
    :return x1: Vector of strong features, values takes either 0 or 1
    :return x2: n x m Matrix of weak features, where n is number_samples and m is number of weak features
    :return m: Mean imputation vector of weak features, 1 x number of weak features
    """
    pairs = pair_seed  # use all of the pair seed
    x1 = list()
    x2 = list()
    for pair in pairs:
        r1 = database.records[pair[0]]
        r2 = database.records[pair[1]]
        c1 = labels_train[next(iter(r1.line_indices))]  # cluster r1 belongs to
        c2 = labels_train[next(iter(r2.line_indices))]  # cluster r2 belongs to
        _x1 = int(c1 == c2)
        x1.append(_x1)
        if database._precomputed_x2:
            _x2 = get_precomputed_x2(database._precomputed_x2, r1, r2)
        else:
            _x2 = get_x2(r1, r2)
        x2.append(_x2)
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
                match = binary_match(f1, f2)
                if not np.isnan(match):
                    match = not match  # Inverting sign, want smaller feature = better match
                                       # (restricting all log reg weights negative)
                x2.append(match)
            elif pairwise_use == 'numerical_difference':
                x2.append(numerical_difference(f1, f2))  # smaller feature = better, OK
            elif pairwise_use == 'number_matches':
                x2.append(np.exp(-number_matches(f1, f2)))  # Using exp(-x) to get smaller feature = better match
            elif pairwise_use == 'special_date_difference':
                x2.append(date_difference(f1, f2))  # smaller feature = better, OK
            elif pairwise_use == 'levenshtein':
                x2.append(np.log(e + levenshtein(f1, f2)))  # smaller feature = better, OK. Using log to decrease weight of large mismatches
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
    Determines the column means of all non-NaN entries in matrix. Replaces NaN values in x
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
    Minimum pairwise distance between two date feature sets, in seconds
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


def levenshtein(feat1, feat2):
    """
    Minimum Levenshtein distance between two sets of strings
    Satisfies ICAR properties
    :param feat1: Set of string features
    :param feat2: Set of string features
    :return: Int or NaN (not enough info to make decision, if either set is empty)
    """
    if bool(feat1) & bool(feat2):
        x = np.Inf
        for s1 in feat1:
            for s2 in feat2:
                x = min(x, Levenshtein.distance(s1, s2))
    else:
        x = np.nan
    return x


def get_precomputed_x2(precomputed_x2, r1, r2):
    """
    Minimum precomputed feature vector
    :param precomputed_x2: Precomputed weak features (smaller valued feature is better)
                           A dict[(id1, id2)] = 1D vector, where id2 >= id1
    :param r1: Record
    :param r2: Record
    :return min_features: 1-D vector with m entries, where m is number of weak features.
    """
    min_features = None
    for line1 in r1.line_indices:
        for line2 in r2.line_indices:
            line_tuple = [line1, line2]
            line_tuple.sort()
            line_tuple = tuple(line_tuple)
            features = precomputed_x2(line_tuple)
            min_features = np.minimum(min_features, features)
    return min_features