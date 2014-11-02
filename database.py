from record import Record, FeatureDescriptor
import numpy as np
from numpy.random import choice
__author__ = 'mbarnes1'


class Synthetic(object):
    """
    Create and corrupt synthetic databases
    """
    def __init__(self, number_entities, records_per_entity, sigma=0):
        """
        :param number_entities:
        :param records_per_entity: Mean number of records per entity
        :param sigma: Standard deviation of records_per_entity
        """
        names = ['One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten']
        types = ['float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float']
        strengths = ['weak', 'weak', 'weak', 'weak', 'weak', 'weak', 'weak', 'weak', 'weak', 'weak']
        blocking = ['', '', '', '', '', '', '', '', '', '']
        pairwise_uses = ['numerical_difference', 'numerical_difference', 'numerical_difference', 'numerical_difference',
                         'numerical_difference', 'numerical_difference', 'numerical_difference', 'numerical_difference',
                         'numerical_difference', 'numerical_difference']
        self.truth = list()
        self.database = Database()
        feature_descriptor = FeatureDescriptor(names, types, strengths, blocking, pairwise_uses)
        self.database.feature_descriptor = feature_descriptor
        record_index = 0
        for entity_index in range(0, number_entities):
            features = np.random.rand(feature_descriptor.number)
            features = features.astype(str)
            if sigma:
                number_records = int(round(np.random.normal(records_per_entity, sigma)))  # number of records for this entity
            else:
                number_records = records_per_entity
            for _ in range(number_records):
                r = Record(record_index, feature_descriptor)
                r.initialize_from_annotation(features)
                self.database.records[record_index] = r
                record_index += 1
                self.truth.append(entity_index)

    #def corrupt(self, ):


class Database(object):
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
    def __init__(self, annotation_path=None, max_records=np.Inf):
        #self._annotation_path = annotation_path
        self.records = dict()
        if annotation_path:
            ins = open(annotation_path, 'r')
            feature_names = next(ins).strip('\n').split(',')  # skip the first line, its a header
            feature_types = next(ins).strip('\n').split(',')  # variable type (e.g. int, string, date)
            ignore_indices = find_in_list(feature_types, '')
            feature_names = remove_indices(ignore_indices, feature_names)
            feature_types = remove_indices(ignore_indices, feature_types)
            feature_strengths = remove_indices(ignore_indices, next(ins).strip('\n').split(','))
            blocking = remove_indices(ignore_indices, next(ins).strip('\n').split(','))
            pairwise_uses = remove_indices(ignore_indices, next(ins).strip('\n').split(','))

            self.feature_descriptor = FeatureDescriptor(feature_names, feature_types, feature_strengths, blocking,
                                                        pairwise_uses)
            # Loop through all the records
            for line_index, sample in enumerate(ins):
                print 'Extracting sample', line_index
                r = Record(line_index, self.feature_descriptor)  # record object from utils
                features = remove_indices(ignore_indices, sample.rstrip('\n').split(','))
                r.initialize_from_annotation(features)
                self.records[line_index] = r
                if line_index >= max_records-1:
                    break
        else:
            self.feature_descriptor = None

    def sample_and_remove(self, number_samples):
        """
        Randomly samples from the database, removes, and returns them as a new database
        :param number_samples: The number of samples to take
        :return new_database: Database created from samples
        """
        line_indices = choice(self.records.keys(), size=number_samples, replace=False)
        new_database = Database()
        new_database.feature_descriptor = self.feature_descriptor
        for line_index in line_indices:
            new_database.records[line_index] = self.records.pop(line_index)
        return new_database

    def dump(self, out_file):
        """
        Writes database to csv file
        :param out_file: String, file name to dump records to
        """
        ins = open(out_file, 'w')
        ins.write(','.join(self.feature_descriptor.names)+'\n')
        ins.write(','.join(self.feature_descriptor.types)+'\n')
        ins.write(','.join(self.feature_descriptor.strengths)+'\n')
        ins.write(','.join(self.feature_descriptor.blocking)+'\n')
        ins.write(','.join(self.feature_descriptor.pairwise_uses)+'\n')
        for counter, (_, r) in enumerate(self.records.iteritems()):
            feature_list = list()
            for subfeatures in r.features:
                feature_list.append(';'.join(map(str, subfeatures)))
            ins.write(','.join(feature_list))
            if counter < len(self.records):  # no line break on final line
                ins.write('\n')
        ins.close()

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


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