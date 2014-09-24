# This is the main utilities file for Traffic Jam
import datetime
from itertools import izip  # Uses iterator instead of list (less memory)


# A single record, i.e. ad or merge of ads
class Record(object):
    """
    This object represents a single ad or entity.
    """
    def __init__(self, ad_number, number_features):
        self.features = list()
        for _ in range(0, number_features):
            self.features.append(set())
        self.ads = {ad_number}

    def merge(self, r2):
        """
        Merges current record in-place with record r2
        :param r2: Record to merge into self
        """
        for feature1, feature2 in izip(self.features, r2.features):
            feature1.update(feature2)

    def initialize_from_annotation(self, features, feature_types):
        """
        Populates record fields from a single line in a csv file, where commas are used to separate different types of
        features and semicolons (;) are used to separate multiple of the same type of feature (e.g. multiple phone
        numbers)
        :param features: list of features
        :param feature_types: Variable type
        """
        if len(self.features) != len(features):
            raise Exception('Feature dimension mismatch')
        if len(self.features) != len(feature_types):
            raise Exception('Feature dimension mismatch')
        index = 0
        for feature, feature_type in izip(features, feature_types):
            sub_features = feature.split(';')
            for f in sub_features:
                if (f != '') & (f != 'none'):
                    if feature_type == 'string':
                        self.features[index].add(f)
                    elif feature_type == 'int':
                        self.features[index].add(int(float(f)))
                    elif feature_type == 'date':
                        self.features[index].add(get_date(f))
            index += 1

    def display(self, names):
        for name, feature in izip(names, self.features):
            print '{0}: {1}'.format(name, feature)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


def get_date(date):
    """
    Gets datetime object from string annotation
    :param date: String in format 'Thu Jan 30 02:41:11 EST 2014'
    :return time: Datetime object, assuming EST
    """
    mo = month(date[4:7])  # month
    d = int(date[8:10])  # day
    h = int(date[11:13])  # hour
    mi = int(date[14:16])  # min
    s = int(date[17:19])  # sec
    z = date[20:23]  # time zone. Assuming EST for now
    y = int(date[24:29])  # year
    time = datetime.datetime(y, mo, d, h, mi, s)
    return time


def month(x):
    month_dict = {
        'Jan': 1,
        'Feb': 2,
        'Mar': 3,
        'Apr': 4,
        'May': 5,
        'Jun': 6,
        'Jul': 7,
        'Aug': 8,
        'Sep': 9,
        'Oct': 10,
        'Nov': 11,
        'Dec': 12,
    }
    return month_dict[x]

