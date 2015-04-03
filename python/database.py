from record import Record, FeatureDescriptor
import numpy as np
from numpy.random import choice
import matplotlib.pyplot as plt
from itertools import izip
__author__ = 'mbarnes1'


class SyntheticDatabase(object):
    """
    Create and corrupt synthetic databases
    """
    def __init__(self, number_entities, records_per_entity, number_features=2):
        """
        Initializes synthetic database
        No initial corruption, so records from the same cluster have exact same features
        :param number_entities:
        :param records_per_entity: Number of records per entity
        :param number_features:
        """
        indices = range(0, number_features)
        names = ['Name_{0}'.format(s) for s in indices]
        types = ['float' for _ in indices]
        strengths = ['weak' for _ in indices]
        blocking = ['' for _ in indices]
        pairwise_uses = ['numerical_difference' for _ in indices]
        self.labels = dict()  # [record identifier, cluster label]
        self.database = Database()
        feature_descriptor = FeatureDescriptor(names, types, strengths, blocking, pairwise_uses)
        self.database.feature_descriptor = feature_descriptor
        record_index = 0
        for entity_index in range(0, number_entities):
            features = np.random.rand(feature_descriptor.number)
            features = features.astype(str)
            number_records = records_per_entity
            for _ in range(number_records):
                r = Record(record_index, feature_descriptor)
                r.initialize_from_annotation(features)
                self.database.records[record_index] = r
                self.labels[record_index] = entity_index
                record_index += 1

    def add(self, number_entities, records_per_entity):
        """
        Adds additional entities to the database
        :param number_entities:
        :param records_per_entity:
        """
        if len(self.labels) != len(self.database.records):
            raise Exception('Number of records and labels do not match')
        current_max_record_id = 0
        current_max_entity_id = 0
        for (record_id, _), (__, entity_id) in izip(self.database.records.iteritems(), self.labels.iteritems()):
            if record_id > current_max_record_id:
                current_max_record_id = record_id
            if entity_id > current_max_entity_id:
                current_max_entity_id = entity_id
        record_index = current_max_record_id+1
        for entity_index in range(current_max_entity_id+1, current_max_entity_id+1+number_entities):
            features = np.random.rand(self.database.feature_descriptor.number)
            features = features.astype(str)
            number_records = records_per_entity
            for _ in range(number_records):
                r = Record(record_index, self.database.feature_descriptor)
                r.initialize_from_annotation(features)
                self.database.records[record_index] = r
                self.labels[record_index] = entity_index
                record_index += 1

    def sample_and_remove(self, number_samples):
        """
        Randomly samles from the database and corresponding labels, removes, and returns them as a new Synthetic object
        :param number_samples: The number of samples to take
        :return new_synthetic: New Synthetic object (includes database and labels)
        """
        new_synthetic = SyntheticDatabase(0, 0)  # empty
        new_synthetic.database = self.database.sample_and_remove(number_samples)
        for key, _ in new_synthetic.database.records.iteritems():
            new_synthetic.labels[key] = self.labels.pop(key)
        return new_synthetic

    def corrupt(self, corruption):
        """
        Added corruption to features
        :param corruption: List of feature corruption vectors for each record
        """
        for corrupt, (_, record) in izip(corruption, self.database.records.iteritems()):
            features = np.array([feature.pop() for feature in record.features])
            features += corrupt
            feature_set = [{feature} for feature in features]
            record.features = feature_set

    def plot(self, labels, title='Feature Distribution', color_seed=None, ax=None):
        """
        Plots 2D features for visualization
        :param labels: Dictionary of [record id, predicted label]
        :param title: String, title for the plot
        :param color_seed: List of colors for each record. Cluster color is color of the earliest record in list. If none, randomly generates colors
        :param ax: Axis to plot on
        """
        if self.database.feature_descriptor.number != 2:
            raise Exception('Can only plot 2D features')
        x1 = list()
        x2 = list()
        label_to_color = dict()
        color_list = list()
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        print '****', title, '****'
        for counter, (idx, record) in enumerate(self.database.records.iteritems()):
            print 'Key:', idx
            label = labels[idx]
            print 'Label', label
            if label not in label_to_color:
                label_to_color[label] = np.random.rand() if color_seed is None else color_seed[counter]
                print 'Added new label color'
            (_x1,) = record.features[0]  # should only be one element in set
            (_x2,) = record.features[1]  # should only be one element in set
            print 'x1:', _x1
            print 'x2:', _x2
            x1.append(_x1)
            x2.append(_x2)
            color_list.append(label_to_color[label])
            print 'Color:', label_to_color[label]
        ax.scatter(x1, x2, s=100, c=color_list, alpha=1.0)
        ax.set_title(title)
        print 'x1:', x1
        print 'x2:', x2
        print 'Colors:', color_list
        ax.axis([-0.2, 1.2, -0.2, 1.2])


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
    def __init__(self, annotation_path=None, max_records=np.Inf, precomputed_x2=None):
        """
        :param annotation_path: String, path to annotation file
        :param max_records: Int, number of records to load from annotation file
        :param precomputed_x2: Precomputed weak features (smaller valued feature is better)
                               A dict[(id1, id2)] = 1D vector, where id2 >= id1
        """
        self.records = dict()
        if annotation_path:
            ins = open(annotation_path, 'r')
            feature_names = next(ins).strip('\n').split(',')  # skip the first line, its a header
            feature_types = next(ins).strip('\n').split(',')  # variable type (e.g. int, string, date)
            ignore_indices = find_in_list(feature_types, 'ignore')
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
        self._precomputed_x2 = precomputed_x2

    def sample_and_remove(self, number_samples):
        """
        Randomly samples from the database, removes, and returns them as a new Database object
        :param number_samples: The number of samples to take
        :return new_database: Database created from samples
        """
        line_indices = choice(self.records.keys(), size=number_samples, replace=False)
        new_database = Database()
        new_database.feature_descriptor = self.feature_descriptor
        for line_index in line_indices:
            new_database.records[line_index] = self.records.pop(line_index)
        return new_database

    def merge(self, labels):
        """
        Merges all records in the database according to the labels
        :param labels: Dict of form [ad_id, cluster_id]
        """
        cluster_to_records = dict()
        for record_id, cluster_id in labels.iteritems():
            if cluster_id in cluster_to_records:
                cluster_to_records[cluster_id].append(record_id)
            else:
                cluster_to_records[cluster_id] = [record_id]
        for _, record_ids in cluster_to_records.iteritems():
            merged_record = None
            for record_id in record_ids:
                if record_id in self.records and merged_record is None:
                    merged_record = self.records[record_id]
                elif record_id in self.records:
                    merged_record.merge(self.records.pop(record_id))

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