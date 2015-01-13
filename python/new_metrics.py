from itertools import izip, combinations
from copy import deepcopy
import cPickle as pickle
from numpy import Inf
__author__ = 'mbarnes1'


class NewMetrics(object):
    def __init__(self, database, entity_resolution):
        """
        Un/semi-supervised entity resolution metrics
        :param database: Reference to Database object
        :param entity_resolution: Reference to EntityResolution object
        """
        print 'Evaluating new metric...'
        self.database = database
        self.er = entity_resolution
        self.net_expected_cost = self.get_net_expected_cost()
        self.greedy_best_cost = self.get_net_greedy_cost('best')
        self.greedy_worst_cost = self.get_net_greedy_cost('worst')
        print 'new metric evaluated.'

    def _get_records(self, entity_index):
        """
        Returns a set of the records comprising entity_index
        :param entity_index: Index of the entity in er to get records from
        :return records: A set of the Record objects
        """
        entity = self.er.entities[entity_index]
        records = set()
        for index in entity.line_indices:
            records.add(self.database.records[index])
        return deepcopy(records)

    def _random_path(self, entity_index):
        """
        Returns the cost of exploring a random path
        :param entity_index: Index of entity in er to explore
        :return path_cost: List of costs when creating the entity
        """
        records = self._get_records(entity_index)
        swooshed, probability_list, strength_list = self.er.rswoosh(records, guarantee_random=True)
        if len(swooshed) != 1:
            print 'Invalid entity found. Dumping results.'
            pickle.dump(self, open('invalid_entity_new_metric.p', 'wb'))
            pickle.dump(records, open('invalid_entity_records.p', 'wb'))
            raise Exception('Invalid entity, no path exists')
        path_cost = list()
        for strength, probability in izip(strength_list, probability_list):
            if strength != 'none':  # if it was a match (valid path)
                path_cost.append(_cost_function(probability))
        return path_cost

    def _expected_path_cost(self, entity_index):
        """
        Uses Monte Carlo sampling to determine the expected path cost of a single entity
        :param entity_index:
        :return expected_path_cost: The expected path cost
        """
        number_monte_carlo = 5  # number of samples to take
        path_costs = list()
        expected_path_cost = 0
        while len(path_costs) < number_monte_carlo:
            path_cost = self._random_path(entity_index)
            path_costs.append(path_cost)
            expected_path_cost += sum(path_cost)
        expected_path_cost /= number_monte_carlo
        return expected_path_cost

    def get_net_expected_cost(self):
        """
        Computes the sum of the expected path cost for each entity
        :return net_cost:
        """
        entity_indices = range(0, len(self.er.entities))
        net_cost = 0
        for entity_index in entity_indices:
            expected_cost = self._expected_path_cost(entity_index)
            net_cost += expected_cost
        return net_cost

    def get_net_greedy_cost(self, greedy_type):
        """
        Computes the sum of the greedy path cost over all entities
        :param greedy_type: string, 'best' or 'worst'
        :return net_cost:
        """
        print 'Evaluating greedy path cost at threshold', self.er._match_function.decision_threshold
        entity_indices = range(0, len(self.er.entities))
        net_cost = 0
        for entity_index in entity_indices:
            expected_cost = self._greedy_cost(entity_index, greedy_type=greedy_type)
            net_cost += expected_cost
        return net_cost

    def _greedy_cost(self, entity_index, greedy_type):
        """
        Find the greedily best or worst merge path cost for a single entity,
        defined as the path with highest or lowest cost at each step
        :param entity_index:
        :param greedy_type: string, 'best' or 'worst'
        :return cost: the greedily best or worst path cost
        """
        if greedy_type not in {'worst', 'best'}:
            raise Exception('Invalid input')
        records_set = self._get_records(entity_index)
        records = {frozenset(r.line_indices): r for r in records_set}

        # Initialize cost dictionary
        print '     Greedily exploring entity merge path with', len(records), 'records'
        print '         Initializing cost dictionary for greedy path exploration...'
        cost_dict = dict()
        all_pairs = combinations([i for i in self.er.entities[entity_index].line_indices], 2)
        for pair in all_pairs:
            record1 = self.database.records[pair[0]]
            record2 = self.database.records[pair[1]]
            match, prob, _ = self.er._match_function.match(record1, record2, 'weak')
            if prob > self.er._match_function.decision_threshold:
                cost_dict[frozenset([frozenset([pair[0]]), frozenset([pair[1]])])] = _cost_function(prob)

        indices1 = None
        indices2 = None
        net_cost = 0
        while cost_dict:
            if greedy_type == 'worst':
                greedy_pair = _max_dict(cost_dict, remove=frozenset([indices1, indices2]))
            else:
                greedy_pair = _min_dict(cost_dict, remove=frozenset([indices1, indices2]))
            net_cost += cost_dict[greedy_pair]
            if len(cost_dict) == 1:
                break
            greedy_pair_list = list(greedy_pair)
            indices1 = greedy_pair_list[0]
            indices2 = greedy_pair_list[1]
            record1 = records.pop(indices1)
            record2 = records.pop(indices2)
            indices = indices1.union(indices2)
            record1.merge(record2)
            for indices_buddy, record_buddy in records.iteritems():
                match, prob, _ = self.er._match_function.match(record1, record_buddy, 'weak')
                if prob > self.er._match_function.decision_threshold:
                    pair = frozenset([indices, indices_buddy])
                    cost_dict[pair] = _cost_function(prob)
            records[indices] = record1
        return net_cost


def _min_dict(dictionary, remove=frozenset()):
    """
    Finds the minimum value in a dictionary and returns the corresponding key
    :param dictionary:
    :param remove: Frozen set. Optionally remove any pairs with these identifiers during the search process
    :return key:
    """
    min_value = Inf
    min_key = None
    keys_to_remove_later = []
    for key, value in dictionary.iteritems():
        if key & remove:  # any intersection?
            keys_to_remove_later.append(key)
        elif value < min_value:
            min_value = value
            min_key = key
    for key in keys_to_remove_later:
        dictionary.pop(key)
    return min_key


def _max_dict(dictionary, remove=frozenset()):
    """
    Finds the maximum value in a dictionary and returns the corresponding key
    :param dictionary:
    :param remove: Frozen set. Optionally remove any pairs with these identifiers during the search process
    :return key:
    """
    max_value = -Inf
    max_key = None
    keys_to_remove_later = []
    for key, value in dictionary.iteritems():
        if key & remove:  # any intersection?
            keys_to_remove_later.append(key)
        elif value > max_value:
            max_value = value
            max_key = key
    for key in keys_to_remove_later:
        dictionary.pop(key)
    return max_key


def _cost_function(prob):
    """
    Given the match probability, returns the cost associated with a merge
    :param prob: The match probability
    :return cost: Resulting cost. Positive = bad. Negative = good.
    """
    cost = (1-prob) - 0.5
    return cost
