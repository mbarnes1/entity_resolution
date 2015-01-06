from itertools import izip
from copy import deepcopy
import cPickle as pickle
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
        number_monte_carlo = 30  # number of samples to take
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

    def get_net_greedy_cost(self, type='worst'):
        """
        Computes the sum of the greedy path cost over all entities
        :param type: string, 'best' or 'worst'
        :return net_cost:
        """
        entity_indices = range(0, len(self.er.entities))
        net_cost = 0
        for entity_index in entity_indices:
            expected_cost = self._greedy_cost(entity_index, type=type)
            net_cost += expected_cost
        return net_cost

    def _greedy_cost(self, entity_index, type='worst'):
        """
        Find the greedily best or worst merge path cost for a single entity,
        defined as the path with highest or lowest cost at each step
        :param entity_index:
        :param type: string, 'best' or 'worst'
        :return cost: the greedily best or worst path cost
        """
        records = self._get_records(entity_index)
        ## Pairwise cost matrix. Merge pair w/ smallest value (above threshold) in matrix
        ## Recompute all pairs w/ merged entity
        ## Repeat process


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



def _cost_function(prob):
    """
    Given the match probability, returns the cost associated with a merge
    :param prob: The match probability
    :return cost: Resulting cost. Positive = bad. Negative = good.
    """
    cost = (1-prob) - 0.5
    return cost
