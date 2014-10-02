from __future__ import division
import cPickle as pickle  # for saving and loading shit, fast
import numpy as np
import multiprocessing
from blocking import BlockingScheme
from pairwise_features import SurrogateMatchFunction
from copy import deepcopy
import gc

__author__ = 'mbarnes1'


# def main():
#     """
#     This creates, runs, evaluates, plots, and saves a Pipeline object.
#     """
#     # User Parameters #
#     number_processes = 50
#     max_block_size = 500  # throw away blocks larger than max_block_size
#     train_size = 5000
#     annotations_path = '/home/scratch/trafficjam/EntityExplorer/Annotations_extended.csv'
#     max_records = np.Inf  # number of records to use from database. Set to np.Inf for all records
#     match_type = 'weak_strong'  # weak, strong, or weak_strong (both) matching in Swoosh
#     ###################
#
#     pipe = EntityResolution(annotations_path, train_size=train_size, match_type=match_type, max_block_size=max_block_size,
#                     max_records=max_records)
#     pipe.run(number_processes)
#     pickle.dump(pipe, open('../../pipe.p', 'wb'))
#     pipe.analyze()
#     pickle.dump(pipe.analysis, open('../../analysis.p', 'wb'))
#     pipe.analysis.print_metrics()
#     print 'Finished saving all results. Attempting to make plots, this is expected to fail on servers.'
#     pipe.analysis.make_plots()


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
            super(EntityResolution.Worker, self).__init__()
            self.jobqueue = jobqueue
            self.resultsqueue = resultsqueue
            self._pipeline = pipeline

        def run(self):
            print 'Worker started'
            for block in iter(self.jobqueue.get, None):
                swooshed = self._pipeline.rswoosh(block)
                if self.resultsqueue.full():
                    print 'Results queue is full!'
                self.resultsqueue.put(swooshed)
            print 'Worker exiting'

    def __init__(self, decision_threshold):
        """
        :param decision_threshold: Threshold for weak feature matching
        :param features_path: String, path to the flat features file
        :param kwargs match_type: Type of matching to perform. String, either 'weak', 'strong', or 'weak_strong'.
        :param kwargs train_size: Int, number of samples to use in training the weak feature classifier
        :param kwargs max_block_size: Int, blocks larger than this are thrown away
        :param kwargs max_records: Int, the number of records to use from features_path. Default all
        """
        # # Required parameters
        # self._features_path = features_path
        #
        # # Optional parameters
        # self.match_type = kwargs.get('match_type', 'weak_strong')
        # self._train_size = kwargs.get('train_size', 5000)
        # self._max_block_size = kwargs.get('max_block_size', 500)
        # self._max_records = kwargs.get('max_records', np.Inf)

        ## Initializations
        self._match_type = None
        self.blocking = None
        self._weak_matches = 0
        self._match_function = SurrogateMatchFunction(decision_threshold)
        self.entities = set()
        self.analysis = None

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

    # def _init_large_files(self):
    #     """
    #     This is a separate initialization function for large files, and those that depend on them.
    #     The reason for a separate function is to initialize large files AFTER creating multiprocessing workers,
    #     otherwise each Python allocates a large amount of memory for new processes
    #     """
    #     #self.database = Database(self._features_path, self._max_records)
    #     self.blocking = BlockingScheme(self.database, self._max_block_size)
    #     #self.surrogate_match_function = SurrogateMatchFunction(self.database, self.blocking.strong_blocks,
    #     #                                                       self._train_size, 0.99)

    def run(self, database_test, match_function, match_type='weak_strong', max_block_size=np.Inf, cores=2):
        """
        This function performs Entity Resolution on all the blocks and merges results to output entities
        :param cores: The number of processes (i.e. workers) to use
        """
        self._match_type = match_type
        self._match_function = match_function
        # Multiprocessing Code
        memory_buffer = pickle.load(open('../../blocking.p', 'rb'))  # so workers are allocated appropriate memory
        jobqueue = multiprocessing.Queue()
        resultsqueue = multiprocessing.Queue()
        workerpool = list()
        for _ in range(cores):
            w = self.Worker(self, jobqueue, resultsqueue)
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
            jobqueue.put(index_block)
        for blockname, indices in self.blocking.weak_blocks.iteritems():
            index_block = set()
            for index in indices:
                index_block.add(records[index])
            jobqueue.put(index_block)
        for _ in workerpool:
            jobqueue.put(None)  # Sentinel objects to allow clean shutdown: 1 per worker.

        # Capture the results
        resultslist = list()
        while len(resultslist) < self.blocking.number_of_blocks():
            resultslist.append(resultsqueue.get())
            print 'Finished', len(resultslist), 'of', self.blocking.number_of_blocks()
        for worker in workerpool:
            worker.join()

        # Convert queue of sets to dictionary for final merge
        swoosheddict = dict()
        counter = 0
        for entities in resultslist:
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
        """
        Inew = set()  # initialize the resolved entities
        while I:  # until entity resolution is complete
            currentrecord = I.pop()  # an arbitrary record
            buddy = False
            for rnew in Inew:  # iterate over Inew
                if self._match_function.match(currentrecord, rnew, self._match_type):
                    buddy = rnew
                    break  # Found a match!
            if buddy:
                currentrecord.merge(buddy)
                I.add(currentrecord)
                Inew.discard(buddy)
            else:
                Inew.add(currentrecord)
        return Inew


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