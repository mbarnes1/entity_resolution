from sklearn import linear_model
import numpy as np
import random
import warnings
from roc import RocCurve
from copy import deepcopy
__author__ = 'mbarnes1'


class SurrogateMatchFunction(object):
    def __init__(self, records, strong_blocks, train_size, decision_threshold):
        self._strong_blocks = deepcopy(strong_blocks)
        self._used_pairs = set()  # pairs used, so as not to repeat samples in training and testing
        self._x1_train = []
        self._x2_train = []
        self.x2_mean = []
        self.roc = []
        self.logreg = linear_model.LogisticRegression()
        self._train(records, train_size)
        self.decision_threshold = decision_threshold

    def _train(self, records, train_size):
        self._x1_train, self._x2_train, self.x2_mean = self._get_pairs(records, train_size, True)
        self.logreg.fit(self._x2_train, self._x1_train)

    def test(self, records, test_size):
        x1_test, x2_test, _ = self._get_pairs(records, test_size, True)
        x1_bar_probability = self.logreg.predict_proba(x2_test)[:, 1]
        output = np.column_stack((x1_test, x1_bar_probability))
        np.savetxt('roc_labels.csv', output, delimiter=",", header='label,probability', fmt='%.1i,%5.5f')
        return RocCurve(x1_test, x1_bar_probability)

    def _get_pairs(self, records, number_samples, balancing):
        # Where balancing is boolean on whether to balance classes using strong_blocks
        numrecords = len(records)
        x1 = list()
        x2 = list()
        pairs = list()
        while len(pairs) < number_samples:
            to_balance = random.choice([True, False])
            if balancing and to_balance:
                while self._strong_blocks:
                    _, block = self._strong_blocks.popitem()
                    if len(block) >= 2:
                        p1 = block.pop()
                        p2 = block.pop()
                        pair = (p1, p2)
                        break
                else:
                    balancing = False
                    warnings.warn('Not enough strong blocks for full balancing')
                    pair = (random.randint(0, numrecords-1), random.randint(0, numrecords-1))
            else:
                pair = (random.randint(0, numrecords-1), random.randint(0, numrecords-1))
            if pair not in self._used_pairs and pair[0] != pair[1]:  # if pair is valid
                self._used_pairs.add(pair)
                pair_flipped = (pair[1], pair[0])
                self._used_pairs.add(pair_flipped)
                pairs.append(pair)
        for pair in pairs:
            r1 = records[pair[0]]
            r2 = records[pair[1]]
            x1.append(get_x1(r1, r2))
            x2.append(get_x2(r1, r2))
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        m = mean_imputation(x2)
        x2 = x2[~np.isnan(x1), :]
        x1 = x1[~np.isnan(x1)]
        return x1, x2, m

    # Determines if two records r1 and r2 match
    def match(self, r1, r2, match_type):
        x1 = get_x1(r1, r2)
        if np.isnan(x1):
            x1 = False
        if match_type == 'strong':
            if x1:
                return True
            else:
                return False
        x2 = get_x2(r1, r2)
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


def mean_imputation(x):
    # Determine the mean
    m = np.nansum(x, axis=0)/np.sum(~np.isnan(x), axis=0)
    m = np.nan_to_num(m)  # if all one feature was NaN, mean will also be NaN. Replace with 0
    # Fill in NaN values. No longer necessary for numpy >= v1.9
    np.copyto(x, m, where=np.isnan(x))
    return m


# Calculates the weak feature vector x2 based on differences between two records
# not using cost yet, complicated formulation
# add restriction types
# correlate date and location to a feasibility
def get_x2(r1, r2):
    x2 = np.zeros(20, dtype=float)  # where 20 is number of features

    # Sites
    x2[0] = exact_matches(r1.sites, r2.sites)
    # Dates
    if bool(r1.dates) & bool(r2.dates):
        for date1 in r1.dates:  # recall dates in a set, don't need to use iteritemes()
            for date2 in r2.dates:
                x2[1] = min(x2[1], abs((date1-date2).seconds))
    else:
        x2[1] = np.nan
    # State
    x2[2] = exact_matches(r1.states, r2.states)
    # City
    x2[3] = exact_matches(r1.cities, r2.cities)
    # Perspective
    x2[4] = 0  # set to zero until I've found an function satisfying ICAR properties
    # Name
    x2[5] = exact_matches(r1.names, r2.names)
    # Age
    x2[6] = difference_minimum(r1.ages, r2.ages)
    # Height
    x2[7] = difference_minimum(r1.heights, r2.heights)
    # Weight
    x2[8] = difference_minimum(r1.weights, r2.weights)
    # Cup size
    x2[9] = exact_matches(r1.cups, r2.cups)
    # Chest size
    x2[10] = difference_minimum(r1.chests, r2.chests)
    # Waist size
    x2[11] = difference_minimum(r1.waists, r2.waists)
    # Hip size
    x2[12] = difference_minimum(r1.hips, r2.hips)
    # Ethnicity
    x2[13] = exact_matches(r1.ethnicities, r2.ethnicities)
    # Skin color
    x2[14] = exact_matches(r1.skincolors, r2.skincolors)
    # Eye color
    x2[15] = exact_matches(r1.eyecolors, r2.eyecolors)
    # Hair color
    x2[16] = exact_matches(r1.haircolors, r2.haircolors)
    # URL
    x2[17] = exact_matches(r1.urls, r2.urls)
    # Media
    x2[18] = exact_matches(r1.medias, r2.medias)
    # Images
    x2[19] = exact_matches(r1.images, r2.images)
    return x2


def get_strong_features(r):
    strong_features = set()
    for phone in r.phones:
        feature = 'phone_' + str(phone)
        strong_features.add(feature)
    for poster in r.posters:
        feature = 'poster_' + str(poster)
        strong_features.add(feature)
    for email in r.emails:
        feature = 'email_' + str(email)
        strong_features.add(feature)
    return strong_features


# Calculates the strong feature vector x1 based on differences between two records
def get_x1(r1, r2):
    # If any phone, poster, or email match, it is a match
    strong_r1 = get_strong_features(r1)
    strong_r2 = get_strong_features(r2)
    if strong_r1 & strong_r2:
        x1 = 1
    elif bool(strong_r1) & bool(strong_r2):
        x1 = 0
    else:
        x1 = np.nan  # not enough values to determine x1
    return x1


# Satisfies ICAR properties
def exact_matches(feat1, feat2):
    if bool(feat1) & bool(feat2):
        x = len(feat1 & feat2)  # intersection
    else:
        x = np.nan
    return x


# Satisfies ICAR properties
def difference_minimum(feat1, feat2):
    if bool(feat1) & bool(feat2):
        x = np.Inf
        for f1 in feat1:
            for f2 in feat2:
                if abs(f1 - f2) < x:
                    x = abs(f1 - f2)
    else:
        x = np.nan
    return x