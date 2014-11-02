from __future__ import division
import cPickle as pickle  # for saving and loading shit, fast
import numpy as np
import multiprocessing
from blocking import BlockingScheme
from pairwise_features import SurrogateMatchFunction
from copy import deepcopy
import gc
from itertools import izip
import traceback
__author__ = 'mbarnes1'


class EntityResolution(object):
    """
    This is the main Pipeline object integrating databases, blocking, Swoosh, and merging.
    """
    class Worker(multiprocessing.Process):
        """
        This is a multiprocessing worker, which when created starts another Python instance.
        After initialization, work begins with .start()
        When finished (determined when sentinel object - None - is queue is processed), clean up with .join()
        """
        def __init__(self, pipeline, jobqueue, resultsqueue):
            """
            :param pipeline: Pipeline object that created this worker
            :param jobqueue: Multiprocessing.Queue() of blocks of records to Swoosh
            :param resultsqueue: Multiprocessing.Queue() of tuples.
                                 tuple[0] = Swoosh results, as set of records
                                 tuple[1] = Decision probabilities, as list of floats in [0, 1]
                                 tuple[2] = Decision type, as list of strings 'strong', 'weak', 'both', or 'none'
            """
            super(EntityResolution.Worker, self).__init__()
            self.jobqueue = jobqueue
            self.resultsqueue = resultsqueue
            self._pipeline = pipeline

        def run(self):
            try:
                self.run_subfunction()
            except:
                print('%s' % (traceback.format_exc()))

        def run_subfunction(self):
            print 'Worker started'
            for block in iter(self.jobqueue.get, None):
                swooshed, decision_list, strength_list = self._pipeline.rswoosh(block)
                self.resultsqueue.put((swooshed, decision_list, strength_list))
            print 'Worker exiting'

    def __init__(self):
        self._match_type = None
        self.blocking = None
        self._match_function = SurrogateMatchFunction()
        self.entities = set()
        self.decision_prob_list = []  # list of probabilities from attempted matches
        self.decision_strength_list = []  # list of decision strengths. 'weak', 'strong', 'weak_strong', or 'none'

    def train(self, database_train, labels_train, train_size, balancing=True):
        """
        Trains the match function used in the entity resolution process
        :param database_train: Database object of samples used for training
        :param labels_train: Dictionary of [line index, cluster label]
        :param train_size: Number of pairs to use in training
        :param balancing: Boolean, whether to balance classes
        :return match_function: Logistic Regression object
        """
        self._match_function.train(database_train, labels_train, train_size, balancing)
        return self._match_function

    def run(self, database_test, match_function, decision_threshold, match_type='weak_strong', max_block_size=np.Inf,
            cores=2):
        """
        This function performs Entity Resolution on all the blocks and merges results to output entities
        :param database_test: Database object to run on
        :param match_function: Handle to surrogate match function object. Must have field decision_threshold and
                               function match(r1, r2, match_type)
        :param decision_threshold: [0, 1] Probability threshold for weak matches
        :param match_type: 'weak', 'strong', or 'weak_strong'. weak_strong matches if weak OR strong features match
        :param max_block_size: Blocks larger than this are thrown away
        :param cores: The number of processes (i.e. workers) to use
        """
        self._match_type = match_type
        self._match_function = match_function
        self._match_function.decision_threshold = decision_threshold
        # Multiprocessing Code
        try:
            memory_buffer = pickle.load(open('../blocking.p', 'rb'))  # so workers are allocated appropriate memory
        except:
            memory_buffer = pickle.load(open('../../blocking.p', 'rb'))  # unit tests
        job_queue = multiprocessing.Queue()
        results_queue = multiprocessing.Queue()
        workerpool = list()
        for _ in range(cores):
            w = self.Worker(self, job_queue, results_queue)
            workerpool.append(w)
        #self._init_large_files()  # initialize large files after creating workers to reduce memory usage by processes
        self.blocking = BlockingScheme(database_test, max_block_size)

        memory_buffer = 0  # so nothing points to memory buffer file
        gc.collect()  # this should delete memory buffer from memory
        for w in workerpool:  # all files should be initialized before starting work
            w.start()
        records = deepcopy(database_test.records)

        # Create block jobs for workers
        for blockname, indices in self.blocking.strong_blocks.iteritems():
            index_block = set()
            for index in indices:
                index_block.add(records[index])
            job_queue.put(index_block)
        for blockname, indices in self.blocking.weak_blocks.iteritems():
            index_block = set()
            for index in indices:
                index_block.add(records[index])
            job_queue.put(index_block)
        for _ in workerpool:
            job_queue.put(None)  # Sentinel objects to allow clean shutdown: 1 per worker.

        # Capture the results
        results_list = list()
        while len(results_list) < self.blocking.number_of_blocks():
            results = results_queue.get()
            results_list.append(results[0])
            self.decision_prob_list += results[1]
            self.decision_strength_list += results[2]
            print 'Finished', len(results_list), 'of', self.blocking.number_of_blocks()
        print 'Joining workers'
        for worker in workerpool:
            worker.join()

        # Convert list of sets to dictionary for final merge
        print 'Converting to dictionary'
        swoosheddict = dict()
        counter = 0
        for entities in results_list:
            while entities:
                entity = entities.pop()
                swoosheddict[counter] = entity
                counter += 1


        """ Single-processing
        results = []
        print 'Processing strong blocks...'
        for _, ads in self.blocking.strong_blocks.iteritems():
            adblock = set()
            for adindex in ads:
                adblock.add(records[adindex])
            swooshed = self.rswoosh(adblock)
            # strong_cluster = fast_strong_cluster_set_wrapper(adblock)
            # if not test_object_set(swooshed, strong_cluster):
            #     raise Exception('Block output does not match')
            results.append(swooshed)
        print 'Processing weak blocks'
        for _, ads in self.blocking.weak_blocks.iteritems():
            adblock = set()
            for adindex in ads:
                adblock.add(records[adindex])
            swooshed = self.rswoosh(adblock)
            # strong_cluster = fast_strong_cluster_set_wrapper(adblock)
            # if not test_object_set(swooshed, strong_cluster):
            #     raise Exception('Block output does not match')
            results.append(swooshed)
        swoosheddict = dict()
        counter = 0
        for entities in results:
            while entities:
                entity = entities.pop()
                swoosheddict[counter] = entity
                counter += 1
        """
        print 'Merging entities...'
        self.entities = merge_duped_records(swoosheddict)
        index_to_cluster = dict()
        for cluster_index, entity in enumerate(self.entities):
            for index in entity.line_indices:
                index_to_cluster[index] = cluster_index
        return index_to_cluster

    # RSwoosh - I and Inew are a set of records, returns a set of records
    def rswoosh(self, I):
        """
        RSwoosh - Benjelloun et al. 2009
        Performs entity resolution on any set of records using merge and match functions
        :param I: Set of input records
        :return Inew: Set of resolved entities (records)
        :return decision_list: List of match and mismatch decision probabilities
        :return strenght_list: List of decision types
        """
        Inew = set()  # initialize the resolved entities
        decision_list = []  # probabilities of each match/mismatch decision
        strength_list = []  # strength of each match. 'weak', 'strong', 'both', or 'none'
        while I:  # until entity resolution is complete
            currentrecord = I.pop()  # an arbitrary record
            buddy = False
            for rnew in Inew:  # iterate over Inew
                match, prob, strength = self._match_function.match(currentrecord, rnew, self._match_type)
                decision_list.append(prob)
                strength_list.append(strength)
                if match:
                    buddy = rnew
                    break  # Found a match!
            if buddy:
                currentrecord.merge(buddy)
                I.add(currentrecord)
                Inew.discard(buddy)
            else:
                Inew.add(currentrecord)
        return Inew, decision_list, strength_list

    def plot_decisions(self):
        """
        This function plots a histogram of decision confidences
        """
        import matplotlib.pyplot as plt
        strong_list = []
        weak_list = []
        both_list = []
        none_list = []

        if len(self.decision_prob_list) != len(self.decision_strength_list):
            raise Exception('List length mismatch')
        for prob, strength in izip(self.decision_prob_list, self.decision_strength_list):
            if strength == 'strong':
                strong_list.append(prob)
            elif strength == 'weak':
                weak_list.append(prob)
            elif strength == 'both':
                both_list.append(prob)
            elif strength == 'none':
                none_list.append(prob)
            else:
                raise Exception('Invalid decision type')
        n, bins, patches = plt.hist([strong_list, weak_list, both_list, none_list], 50, histtype='stepfilled',
                                    stacked=True) #label=['Strong', 'Weak', 'Both', 'None'])
        plt.legend()
        #plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
        plt.xlabel('Match Probability')
        plt.ylabel('Occurences')
        plt.title('Match Probability Distribution')
        plt.show()
        #except:
         #   print 'Cannot make plots on server'


def merge_duped_records(tomerge):
    """
    Merges any entities containing the same ad.
    This is necessary due to placing records into multiple blocks.
    :param tomerge: Dictionary of [integer index, merged record]
    :return: entities: set of records (the final entities)
    """
    # Input: dictionary of [key, merged record]
    # Create dual hash tables
    swoosh2indices = dict()  # [swoosh index, set of ad indices]  node --> edges
    index2swoosh = dict()  # [ad index, list of swoosh indices]  edge --> nodes (note an edge can lead to multiple nodes)
    for s, record in tomerge.iteritems():
        indices = record.line_indices
        swoosh2indices[s] = indices
        for index in indices:
            if index in index2swoosh:
                index2swoosh[index].append(s)
            else:
                index2swoosh[index] = [s]

    # Determine connected components
    explored_swooshed = set()  # swooshed records already explored
    explored_indices = set()
    toexplore = list()  # ads to explore
    entities = set()  # final output of merged records
    counter = 1
    for s, indices in swoosh2indices.iteritems():  # iterate over all the nodes
        if s not in explored_swooshed:  # was this node already touched?
            explored_swooshed.add(s)
            cc = set()  # Its a new connected component!
            cc.add(s)  # first entry
            toexplore.extend(list(indices))  # explore all the edges!
            while toexplore:  # until all paths have been explored
                index = toexplore.pop()  # explore an edge!
                if index not in explored_indices:
                    explored_indices.add(index)
                    connectedswooshes = index2swoosh[index]  # the nodes this edge leads to
                    for connectedswoosh in connectedswooshes:
                        if connectedswoosh not in explored_swooshed:  # was this node already touched?
                            cc.add(connectedswoosh)  # add it to the connected component
                            explored_swooshed.add(connectedswoosh)  # add it to the touched nodes list
                            toexplore.extend(list(swoosh2indices[connectedswoosh]))  # explore all this node's edges

            # Found complete connected component, now do the actual merging
            merged = tomerge[cc.pop()]
            for c in cc:
                merged.merge(tomerge[c])
            entities.add(merged)
        counter += 1
    return entities