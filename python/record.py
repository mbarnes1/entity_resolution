# This is the main utilities file for Traffic Jam
import datetime
from itertools import izip  # Uses iterator instead of list (less memory)


# A single record, i.e. ad or merge of ads
class Record(object):
    """
    This object represents a single ad or entity.
    """
    def __init__(self, line_index=0, feature_descriptor=0):
        """
        :param line_index: Unique ad number for identification
        :param feature_descriptor: FeatureDescriptor object
        """
        self.feature_descriptor = feature_descriptor
        self.features = list()
        for _ in range(0, feature_descriptor.number):
            self.features.append(set())
        self.line_indices = {line_index}

    def merge(self, r2):
        """
        Merges current record in-place with record r2
        :param r2: Record to merge into self
        """
        for feature1, feature2 in izip(self.features, r2.features):
            feature1.update(feature2)
        self.line_indices.update(r2.line_indices)

    def initialize_from_annotation(self, features):
        """
        Populates record fields from a single line in a csv file, where commas are used to separate different types of
        features and semicolons (;) are used to separate multiple of the same type of feature (e.g. multiple phone
        numbers)
        :param features: list of features. Each feature is a string
        """
        if len(self.features) != len(features):
            raise Exception('Feature dimension mismatch')
        if len(self.features) != len(self.feature_descriptor.types):
            raise Exception('Feature dimension mismatch')
        index = 0
        for feature, feature_type in izip(features, self.feature_descriptor.types):
            sub_features = feature.split(';')
            for f in sub_features:
                if (f != '') & (f != 'none'):
                    if feature_type == 'string':
                        self.features[index].add(f)
                    elif feature_type == 'int':
                        self.features[index].add(int(float(f)))
                    elif feature_type == 'float':
                        self.features[index].add(float(f))
                    elif feature_type == 'date':
                        self.features[index].add(get_date(f))
            index += 1

    def get_features(self, filter_strength):
        """
        Returns all the features from a record of specified strength
        :param record: Record object
        :param filter_strength: Strength of feature to return, either 'weak' or 'strong'
        :return features: Set of all the requested features as strings of form 'feature_name + _ + feature'
        """
        features = set()
        for feature, name, strength in izip(self.features, self.feature_descriptor.names, self.feature_descriptor.strengths):
            if strength == filter_strength:
                for subfeature in feature:
                    features.add(name + '_' + str(subfeature))
        return features

    def display(self, names):
        for name, feature in izip(names, self.features):
            print '{0}: {1}'.format(name, feature)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class FeatureDescriptor(object):
    """
    This object contains descriptions of features
    """
    def __init__(self, names, types, strengths, blocking, pairwise_uses):
        self.names = names
        self.types = types
        self.strengths = strengths
        self.blocking = blocking
        self.pairwise_uses = pairwise_uses
        self.number = len(self.names)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


def get_date(date):
    """
    Gets datetime object from string annotation
    :param date: String in format 'Thu Jan 30 02:41:11 EST 2014'
    :return time: Datetime object, assuming EST
    """
    y = int(date[0:4])
    mo = int(date[5:7])
    d = int(date[8:10])
    h = int(date[11:13])
    mi = int(date[14:16])
    s = int(date[17:19])
    time = datetime.datetime(y, mo, d, h, mi, s)
    return time

