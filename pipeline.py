from __future__ import division
import cPickle as pickle  # for saving and loading shit, fast
import numpy as np
import multiprocessing
from database import RecordDatabase
from blocking import BlockingScheme
from pairwise_features import SurrogateMatchFunction
from analysis import Analysis
from copy import deepcopy
import gc

__author__ = 'mbarnes1'


def main():
    """
    This creates, runs, evaluates, plots, and saves a Pipeline object.
    """
    # User Parameters #
    number_processes = 50
    max_block_size = 500  # throw away blocks larger than max_block_size
    train_size = 5000
    annotations_path = '/home/scratch/trafficjam/EntityExplorer/Annotations_extended.csv'
    max_records = np.Inf  # number of records to use from database. Set to np.Inf for all records
    match_type = 'weak_strong'  # weak, strong, or weak_strong (both) matching in Swoosh
    ###################

    pipe = Pipeline(annotations_path, train_size=train_size, match_type=match_type, max_block_size=max_block_size,
                    max_records=max_records)
    pipe.run(number_processes)
    pickle.dump(pipe, open('../../pipe.p', 'wb'))
    pipe.analyze()
    pickle.dump(pipe.analysis, open('../../analysis.p', 'wb'))
    pipe.analysis.print_metrics()
    print 'Finished saving all results. Attempting to make plots, this is expected to fail on servers.'
    pipe.analysis.make_plots()


class Pipeline(object):
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
            super(Pipeline.Worker, self).__init__()
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

    def __init__(self, features_path, **kwargs):
        """
        :param features_path: String, path to the flat features file
        :param kwargs match_type: Type of matching to perform. String, either 'weak', 'strong', or 'weak_strong'.
        :param kwargs train_size: Int, number of samples to use in training the weak feature classifier
        :param kwargs max_block_size: Int, blocks larger than this are thrown away
        :param kwargs max_records: Int, the number of records to use from features_path. Default all
        """
        # Required parameters
        self._features_path = features_path

        # Optional parameters
        self.match_type = kwargs.get('match_type', 'weak_strong')
        self._train_size = kwargs.get('train_size', 5000)
        self._max_block_size = kwargs.get('max_block_size', 500)
        self._max_records = kwargs.get('max_records', np.Inf)

        ## Initializations
        self.database = []
        self.blocking = []
        self._weak_matches = 0
        self.strong_clusters = []
        self.surrogate_match_function = []
        self.entities = set()
        self.analysis = []

    def _init_large_files(self):
        """
        This is a separate initialization function for large files, and those that depend on them.
        The reason for a separate function is to initialize large files AFTER creating multiprocessing workers,
        otherwise each Python allocates a large amount of memory for new processes
        """
        self.database = RecordDatabase(self._features_path, self._max_records)
        self.blocking = BlockingScheme(self.database, self._max_block_size)
        self.strong_clusters = fast_strong_cluster(self.database.records)
        self.surrogate_match_function = SurrogateMatchFunction(self.database, self.blocking.strong_blocks,
                                                               self._train_size, 0.99)

    def analyze(self):
        self.analysis = Analysis(self)

    def run(self, cores=2):
        """
        This function performs Entity Resolution on all the blocks and merges results to output entities
        :param cores: The number of processes (i.e. workers) to use
        """
        # Multiprocessing Code
        memory_buffer = pickle.load(open('../../blocking.p', 'rb'))  # so workers are allocated appropriate memory
        jobqueue = multiprocessing.Queue()
        resultsqueue = multiprocessing.Queue()
        workerpool = list()
        for _ in range(cores):
            w = self.Worker(self, jobqueue, resultsqueue)
            workerpool.append(w)
        self._init_large_files()  # initialize large files after creating workers to reduce memory usage by processes
        memory_buffer = 0  # so nothing points to memory buffer file
        gc.collect()  # this should delete memory buffer from memory
        for w in workerpool:  # all files should be initialized before starting work
            w.start()
        records = deepcopy(self.database.records)

        # Create block jobs for workers
        for blockname, ads in self.blocking.strong_blocks.iteritems():
            adblock = set()
            for adindex in ads:
                adblock.add(records[adindex])
            jobqueue.put(adblock)
        for blockname, ads in self.blocking.weak_blocks.iteritems():
            adblock = set()
            for adindex in ads:
                adblock.add(records[adindex])
            jobqueue.put(adblock)
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
                if self.surrogate_match_function.match(currentrecord, rnew, self.match_type):
                    buddy = rnew
                    break  # Found a match!
            if buddy:
                currentrecord.merge(buddy)
                I.add(currentrecord)
                Inew.discard(buddy)
            else:
                Inew.add(currentrecord)
        return Inew


def test_object_set(set1, set2_destructive):
    """
    Tests whether two sets of record objects are equal. Used for debugging
    :param set1: set of record objects
    :param set2_destructive: set of record objects, which would be modified w/o deepcopy operation
    """
    set2 = deepcopy(set2_destructive)
    if len(set1) != len(set2):
        return False
    for obj1 in set1:
        for obj2 in set2:
            if obj1 == obj2:
                set2.discard(obj2)
                break
        else:  # obj1 not in set2
            return False
    return True


def fast_strong_cluster_set_wrapper(records):
    """
    This is a wrapper function for fast_strong_cluster for set inputs, instead of dictionary inputs
    :param records: A set of record objects
    :return strong_cluster: A set of record objects, the resolved entities
    """
    record_dictionary = {}
    for record in records:
        if len(record.ads) > 1:
            raise Exception('Record ads must be unique')
        else:
            ad = deepcopy(record.ads)
            ad = ad.pop()
            record_dictionary[ad] = record
    strong_cluster = fast_strong_cluster(record_dictionary)
    return strong_cluster


def fast_strong_cluster(records):
    """
    Merges records with any strong features in common.
    Equivalent to running RSwoosh using only strong matches, except much faster because of explicit graph exploration
    :param records: Dictionary of [ad_number, record object]
    :return entities: Set of records, the resolved entities
    """
    records_copy = deepcopy(records)  # prevents input mutation
    strong2ad = dict()  # [swoosh index, set of ad indices]  node --> edges
    ad2strong = dict()  # [ad index, list of swoosh indices]  edge --> nodes (note an edge can lead to multiple nodes)
    entities = set()  # final output of merged records
    for _, record in records_copy.iteritems():
        ads = record.ads
        strong_features = record.get_features('strong')
        if not strong_features:  # no strong features, insert singular entity
            entities.add(record)
        for strong in strong_features:
            if strong in strong2ad:
                strong2ad[strong].extend(list(ads))
            else:
                strong2ad[strong] = list(ads)
            for ad in ads:
                if ad in ad2strong:
                    ad2strong[ad].append(strong)
                else:
                    ad2strong[ad] = [strong]

    # Determine connected components
    explored_strong = set()  # swooshed records already explored
    explored_ads = set()
    toexplore = list()  # ads to explore
    counter = 1
    for ad, strong_features in ad2strong.iteritems():
        if ad not in explored_ads:
            explored_ads.add(ad)
            cc = set()
            cc.add(ad)
            toexplore.extend(list(strong_features))
            while toexplore:
                strong = toexplore.pop()
                if strong not in explored_strong:
                    explored_strong.add(strong)
                    connected_ads = strong2ad[strong]
                    for connected_ad in connected_ads:
                        if connected_ad not in explored_ads:
                            cc.add(connected_ad)
                            explored_ads.add(connected_ad)
                            toexplore.extend(list(ad2strong[connected_ad]))

            # Found complete connected component, now do the actual merging
            merged = records_copy[cc.pop()]
            for c in cc:
                merged.merge(records_copy[c])
            entities.add(merged)
        counter += 1
    return entities


def merge_duped_records(tomerge):
    """
    Merges any entities containing the same ad.
    This is necessary due to placing records into multiple blocks.
    :param tomerge: Dictionary of [integer index, merged record]
    :return: entities: set of records (the final entities)
    """
    # Input: dictionary of [key, merged record]
    # Create dual hash tables
    swoosh2ad = dict()  # [swoosh index, set of ad indices]  node --> edges
    ad2swoosh = dict()  # [ad index, list of swoosh indices]  edge --> nodes (note an edge can lead to multiple nodes)
    for s, record in tomerge.iteritems():
        ads = record.ads
        swoosh2ad[s] = ads
        for ad in ads:
            if ad in ad2swoosh:
                ad2swoosh[ad].append(s)
            else:
                ad2swoosh[ad] = [s]

    # Determine connected components
    explored_swooshed = set()  # swooshed records already explored
    explored_ads = set()
    toexplore = list()  # ads to explore
    entities = set()  # final output of merged records
    counter = 1
    for s, ads in swoosh2ad.iteritems():  # iterate over all the nodes
        if s not in explored_swooshed:  # was this node already touched?
            explored_swooshed.add(s)
            cc = set()  # Its a new connected component!
            cc.add(s)  # first entry
            toexplore.extend(list(ads))  # explore all the edges!
            while toexplore:  # until all paths have been explored
                ad = toexplore.pop()  # explore an edge!
                if ad not in explored_ads:
                    explored_ads.add(ad)
                    connectedswooshes = ad2swoosh[ad]  # the nodes this edge leads to
                    for connectedswoosh in connectedswooshes:
                        if connectedswoosh not in explored_swooshed:  # was this node already touched?
                            cc.add(connectedswoosh)  # add it to the connected component
                            explored_swooshed.add(connectedswoosh)  # add it to the touched nodes list
                            toexplore.extend(list(swoosh2ad[connectedswoosh]))  # explore all this node's edges

            # Found complete connected component, now do the actual merging
            merged = tomerge[cc.pop()]
            for c in cc:
                merged.merge(tomerge[c])
            entities.add(merged)
        counter += 1
    return entities


if __name__ == '__main__':
    main()