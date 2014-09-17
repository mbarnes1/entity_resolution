#/usr/bin/python

import numpy as np
from itertools import permutations
__author__ = 'mbarnes1'


class Analysis(object):
    """
    This object performs all Entity Resolution analysis
    """
    def __init__(self, pipeline):
        self.number_records = len(pipeline.database.records)
        print 'Evaluating surrogate match function...'
        self.surrogate_roc = pipeline.surrogate_match_function.test(pipeline.database.records, pipeline._train_size)
        print 'Evaluating blocks...'
        self.block_sizes = _get_block_sizes(pipeline.blocking.strong_blocks, pipeline.blocking.weak_blocks)
        self.number_blocks = pipeline.blocking.number_of_blocks()
        print 'Evaluating entities...'
        self.entity_sizes = _get_entity_sizes(pipeline.entities)
        self.number_entities = len(pipeline.entities)
        print 'Evaluating strong clusters...'
        self.strong_cluster_sizes = _get_entity_sizes(pipeline.strong_clusters)
        self.number_strong_clusters = len(pipeline.strong_clusters)
        pairs_entities = get_pairs(pipeline.entities)
        pairs_truth = get_pairs(pipeline.strong_clusters)
        try:
            self.pair_precision = pair_precision(pairs_entities, pairs_truth)
            self.pair_recall = pair_recall(pairs_entities, pairs_truth)
            self.pair_f1 = 2*self.pair_precision*self.pair_recall/(self.pair_precision + self.pair_recall)
        except ZeroDivisionError:
            self.pair_precision = np.NaN
            self.pair_recall = np.NaN
            self.pair_f1 = np.NaN

    def make_plots(self):
        """
        Make weak feature classifier ROC curve, strong entity sizes, entity sizes, and block sizes figures
        Tries to catch error for when running on server, but it does not work
        """
        try:
            import matplotlib.pyplot as plt
            make_plots = True
        except ImportError or RuntimeError:
            make_plots = False
        if make_plots:
            ## Surrogate ROC
            self.surrogate_roc.make_plot(True)

            ## Strong Cluster Sizes
            plt.plot(self.strong_cluster_sizes[:, 0], self.strong_cluster_sizes[:, 1], 'ro')
            plt.xlabel('Number of ads in strong cluster')
            plt.ylabel('Occurences')
            plt.title('Strong Cluster Sizes')
            plt.show()

            ## Entity Sizes
            plt.plot(self.entity_sizes[:, 0], self.entity_sizes[:, 1], 'ro')
            plt.xlabel('Number of ads in entity')
            plt.ylabel('Occurences')
            plt.title('Entity Sizes')
            plt.show()

            ## Block Sizes
            plt.plot(self.block_sizes[:, 0], self.block_sizes[:, 1], 'ro')
            plt.xlabel('Number of ads in block')
            plt.ylabel('Occurences')
            plt.title('Block Sizes')
            plt.show()

    def print_metrics(self):
        print 'Number of records {0}'.format(self.number_records)
        print 'Number of blocks {0}'.format(self.number_blocks)
        print 'Number of entities {0}'.format(self.number_entities)
        print 'Pairwise Precision: {:.3}'.format(self.pair_precision)
        print 'Pairwise Recall: {:.3}'.format(self.pair_recall)
        print 'Pairwise F_1: {:.2}'.format(self.pair_f1)


def pair_precision(pairs_entities, pairs_truth):
    precision = float(len(pairs_entities & pairs_truth))/len(pairs_entities)
    return precision


def pair_recall(pairs_entities, pairs_truth):
    recall = float(len(pairs_entities & pairs_truth))/len(pairs_truth)
    return recall


def get_pairs(entities):
    """
    Returns all pairs in entities (i.e. ER output), for evaluation
    :param entities: A set of records (entities)
    :return pairs: A set of pair tuples. Permutations, so both directions included -- (a,b) and (b,a)
    """
    pairs = set()
    counter = 1
    max_counter = len(entities)
    for entity in entities:
        counter += 1
        ads = entity.ads
        print 'Getting', len(ads)*(len(ads)-1), 'pairs from entity/cluster', counter, 'of', max_counter, '.', len(pairs), 'cumulative pairs.'
        pairs.update(set(permutations(ads, 2)))
    return pairs


def _get_block_sizes(strong_blocks, weak_blocks):
    """
    Determines the block sizes
    :param strong_blocks: A dictionary of strong blocks. [block_name, list of ad indices]
    :param weak_blocks: A dictionary of weak blocks. [block_name, list of ad indices]
    :return: number_bins x 2 array of block sizes. First column is sorted size, second column is number of occurences
    """
    block_sizes = dict()
    for _, ads in strong_blocks.iteritems():
        size = len(ads)
        if size in block_sizes:
            block_sizes[size] += 1
        else:
            block_sizes[size] = 1
    for _, ads in weak_blocks.iteritems():
        size = len(ads)
        if size in block_sizes:
            block_sizes[size] += 1
        else:
            block_sizes[size] = 1
    counter = 0
    x = np.empty(len(block_sizes),)
    y = np.empty(len(block_sizes),)
    for size, occurences in block_sizes.iteritems():
        x[counter] = size
        y[counter] = occurences
        counter += 1
    xy = np.vstack((x, y)).T
    xy = xy[xy[:, 0].argsort()]
    return xy


def _get_entity_sizes(entities):
    """
    Determines the entity sizes, for plotting
    :param entities: A set of records
    :return: number_bins x 2 array of entity sizes. First column is sorted size, second column is number of occurences
    """
    entity_sizes = dict()
    for entity in entities:
        size = len(entity.ads)
        if size in entity_sizes:
            entity_sizes[size] += 1
        else:
            entity_sizes[size] = 1
    counter = 0
    x = np.empty(len(entity_sizes),)
    y = np.empty(len(entity_sizes),)
    for size, occurences in entity_sizes.iteritems():
        x[counter] = size
        y[counter] = occurences
        counter += 1
    xy = np.vstack((x, y)).T
    xy = xy[xy[:, 0].argsort()]
    return xy