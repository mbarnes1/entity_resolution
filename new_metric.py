from itertools import izip
__author__ = 'mbarnes1'


class NewMetric(object):
    def __init__(self, database, entity_resolution):
        """
        Un/semi-supervised entity resolution metrics
        :param database: Reference to Database object
        :param entity_resolution: Reference to EntityResolution object
        """
        self.database = database
        self.er = entity_resolution

    def _get_records(self, entity_index):
        """
        Returns a set of the records comprising entity_index
        :param entity_index: Index of the entity in er to get records from
        :return records: A set of the Record objects
        """
        entity = self.er.entities[entity_index]
        records = set()
        for id in entity.line_indices:
            records.add(self.database.records[id])
        return records

    def _random_path(self, entity_index):
        """
        Returns the cost of exploring a random path
        :param entity_index: Index of entity in er to explore
        :return path_cost: List of costs when creating the entity
        """
        records = self._get_records(entity_index)
        swooshed, probability_list, strength_list = self.er.rswoosh(records, guarantee_random=True)
        if len(swooshed) != 1:
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

    def net_expected_cost(self):
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


def _cost_function(prob):
    """
    Given the match probability, returns the cost associated with a merge
    :param prob: The match probability
    :return cost: Resulting cost. Positive = bad. Negative = good.
    """
    cost = (1-prob) - 0.5
    return cost
