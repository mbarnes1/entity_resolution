"""
This is the blocking scheme used to make Entity Resolution computationally feasible.
"""
import numpy as np
from itertools import izip
from copy import copy
__author__ = 'mbarnes1'


class BlockingScheme(object):
    def __init__(self, database, max_block_size=np.Inf, single_block=False):
        """
        :param database: RecordDatabase object
        :param max_block_size: Integer. Blocks larger than this are thrown away (not informative & slow to process))
        :param single_block: Boolean, if True puts all records into a single weak block
        """
        self._max_block_size = max_block_size
        self.strong_blocks = dict()
        self.weak_blocks = dict()
        if not single_block:
            self._generate_strong_blocks(database)
            self._generate_weak_blocks(database)
            self._clean_blocks()
            self._complete_blocks(database.records.keys())
        else:
            self._max_block_size = np.Inf
            self.weak_blocks['All'] = set(database.records.keys())

    # Inserts ads that were not blocked or whose block was thrown away
    def _complete_blocks(self, keys):
        """
        Finds ads missing from blocking scheme (due to sparse features), and ads them as single ads to weak blocks
        :param keys: List of all the record identifiers that should be in the clustering
        """
        used_ads = set()
        for _, ads in self.strong_blocks.iteritems():
            used_ads.update(ads)
        for _, ads in self.weak_blocks.iteritems():
            used_ads.update(ads)
        missing_ads = set(keys)
        missing_ads.difference_update(used_ads)
        for ad in missing_ads:
            block = 'singular_ad_' + str(ad)
            self.weak_blocks[block] = {ad}

    def _clean_blocks(self):
        """
        Removed blocks larger than max_block_size
        """
        toremove = list()
        for block, ads in self.strong_blocks.iteritems():
            blocksize = len(ads)
            if blocksize > self._max_block_size:
                toremove.append(block)
        for remove in toremove:
            del self.strong_blocks[remove]

        toremove = list()
        for block, ads in self.weak_blocks.iteritems():
            blocksize = len(ads)
            if blocksize > self._max_block_size:
                toremove.append(block)
        for remove in toremove:
            del self.weak_blocks[remove]

    def number_of_blocks(self):
        """
        Determines the total number of blocks
        :return num_blocks: The total number of weak and strong blocks
        """
        num_blocks = len(self.weak_blocks) + len(self.strong_blocks)
        return num_blocks

    def _generate_strong_blocks(self, database):
        self._generate_blocks('strong', self.strong_blocks, database)

    def _generate_weak_blocks(self, database):
        self._generate_blocks('weak', self.weak_blocks, database)

    @staticmethod
    def _generate_blocks(block_strength, blocks_pointer, database):
        """
        Generates the blocking scheme [block_name, set of ad indices in block]
        :param block_strength: String 'weak' or 'strong'
        :param blocks_pointer: Blocks to mutate, either self.strong_blocks or self.weak_blocks
        :param database: RecordDatabase object
        """
        to_block = list()
        for index, (strength, blocking) in enumerate(izip(database.feature_descriptor.strengths,
                                                          database.feature_descriptor.blocking)):
                if (strength == block_strength) & (blocking == 'block'):  # did user specify blocking for this feature?
                    to_block.append(index)
        for record_id, record in database.records.iteritems():  # loop through all the records
            print block_strength, 'blocking ad', record_id
            for index in to_block:
                feature_name = database.feature_descriptor.names[index]
                for subfeature in record.features[index]:
                    feature = feature_name + '_' + str(subfeature)
                    if feature in blocks_pointer:
                        blocks_pointer[feature].add(record_id)
                    else:
                        blocks_pointer[feature] = {record_id}