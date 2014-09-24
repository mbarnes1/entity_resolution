from record import Record
import numpy as np
__author__ = 'mbarnes1'


class RecordDatabase(object):
    """
    A collection of record objects, using a dictionary with [index, record object]
    Reads from a flat file of the format:
    FeatureName1, FeatureName2, ..., FeatureNameN
    FeatureType1, FeatureType2, ..., FeatureTypeN
    FeatureStrength1, FeatureStrength2, ..., FeatureStrengthN
    Blocking1, Blocking2, ..., BlockingN
    PairwiseUse1, PairwiseUse2, ..., PairwiseUseN
    Ad1_Feature1, Ad1_Feature2, ..., Ad1_FeatureN
    .................
    AdM_Feature1, AdM_Feature2, ..., AdM_FeatureN
    
    where M is the number of ads and N is the number of features
    """
    def __init__(self, annotation_path, max_records=np.Inf):
        self._annotation_path = annotation_path
        self.records = dict()
        self._max_records = max_records
        ins = open(self._annotation_path, 'r')
        feature_names = next(ins).strip('\n').split(',')  # skip the first line, its a header
        feature_types = next(ins).strip('\n').split(',')  # variable type (e.g. int, string, date)
        ignore_indices = find_in_list(feature_types, '')
        self.feature_names = remove_indices(ignore_indices, feature_names)
        self.feature_types = remove_indices(ignore_indices, feature_types)
        self.feature_strengths = remove_indices(ignore_indices, next(ins).strip('\n').split(','))
        self.blocking = remove_indices(ignore_indices, next(ins).strip('\n').split(','))
        self.pairwise_uses = remove_indices(ignore_indices, next(ins).strip('\n').split(','))

        # Loop through all the records
        for ad_number, sample in enumerate(ins):
            print 'Extracting sample', ad_number
            r = Record(ad_number, len(self.feature_names))  # record object from utils
            features = remove_indices(ignore_indices, sample.rstrip('\n').split(','))
            r.initialize_from_annotation(features, self.feature_types)
            self.records[ad_number] = r
            if ad_number >= self._max_records-1:
                break

    # def get_features(self, record, feature_indices):
    #     """
    #     Returns all the requested features from a record
    #     :param record: Record object
    #     :return features: Set of all the requested features as strings of form 'feature_name + _ + feature'
    #     """
    #     features = set()
    #     for feature_index in feature_indices:
    #         feature = record.features[feature_index]
    #         feature_name = self.feature_names[feature_index]
    #         for subfeature in feature:
    #             features.add(feature_name + '_' + str(subfeature))
    #     return features


def remove_indices(remove, lst):
    """
    Removes indices from a list
    :param remove: A list of indices to remove from lst
    :param lst: List to remove entries from
    :return new_lst: A new list with entries from lst removed
    """
    new_lst = [item for i, item in enumerate(lst) if i not in remove]
    return new_lst


def find_in_list(lst, field):
    """
    Finds all occurences of field in list and returns their indicies
    :param field: The entry to find in list
    :param lst: List to search
    :return i: Indices of entries in lst
    """
    return [i for i, x in enumerate(lst) if x == field]