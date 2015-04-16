from __future__ import division
import cPickle as pickle  # for saving and loading shit, fast
import multiprocessing
from pairwise_features import get_weak_pairwise_features
import gc
import traceback
import os
from itertools import permutations, combinations_with_replacement
import networkx
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
                swooshed = self._pipeline.rswoosh(block)
                self.resultsqueue.put(swooshed)
            print 'Worker exiting'

    def __init__(self):
        print 'Entity resolution initialized'

    def run(self, database_test, match_function, blocking_scheme, cores=1):
        """
        This function performs Entity Resolution on all the blocks and merges results to output entities
        :param database_test: Database object to run on. Will modify in place.
        :param match_function: Handle to surrogate match function object. Must have field decision_threshold and
                               function match(r1, r2, match_type)
        :param blocking_scheme: BlockingScheme object
        :param cores: The number of processes (i.e. workers) to use
        :return identifier_to_cluster: Predicted labels, of dictionary form [identifier, cluster label]
        """
        if match_function.ICAR is False:
            raise Exception('Match function must satsify ICAR properties for R-Swoosh')
        self._match_function = match_function
        # Multiprocessing code
        if cores > 1:  # large job, use memory bufffer
            this_path = os.path.dirname(os.path.realpath(__file__))
            buffer_path = this_path + '/memory_buffer.p'
            memory_buffer = pickle.load(open(buffer_path, 'rb'))  # so workers are allocated appropriate memory
        job_queue = multiprocessing.Queue()
        results_queue = multiprocessing.Queue()
        workerpool = list()
        for _ in range(cores):
            w = self.Worker(self, job_queue, results_queue)
            workerpool.append(w)
        #self._init_large_files()  # initialize large files after creating workers to reduce memory usage by processes

        memory_buffer = 0  # so nothing points to memory buffer file
        if cores > 1:
            gc.collect()  # this should delete memory buffer from memory
        for w in workerpool:  # all files should be initialized before starting work
            w.start()
        records = database_test.records

        # Create block jobs for workers
        for blockname, indices in blocking_scheme.strong_blocks.iteritems():
            index_block = set()
            for identifier in indices:
                index_block.add(records[identifier])
            job_queue.put(index_block)
        for blockname, indices in blocking_scheme.weak_blocks.iteritems():
            index_block = set()
            for identifier in indices:
                index_block.add(records[identifier])
            job_queue.put(index_block)
        for _ in workerpool:
            job_queue.put(None)  # Sentinel objects to allow clean shutdown: 1 per worker.

        # Capture the results
        results_list = list()
        while len(results_list) < blocking_scheme.number_of_blocks():
            results = results_queue.get()
            results_list.append(results)
            print 'Finished', len(results_list), 'of', blocking_scheme.number_of_blocks()
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
        entities = merge_duped_records(swoosheddict)
        identifier_to_cluster = dict()
        for cluster_index, entity in enumerate(entities):
            for identifier in entity.line_indices:
                identifier_to_cluster[identifier] = cluster_index
        print 'entities merged.'
        return identifier_to_cluster

    def rswoosh(self, I):
        """
        RSwoosh - Benjelloun et al. 2009
        Performs entity resolution on any set of records using merge and match functions
        :param I: Set of input records
        :return Inew: Set of resolved entities (records)
        """
        Inew = set()  # initialize the resolved entities
        while I:  # until entity resolution is complete
            currentrecord = I.pop()  # an arbitrary record
            buddy = False
            for rnew in Inew:  # iterate over Inew
                match, prob = self._match_function.match(currentrecord, rnew)
                if match:
                    buddy = rnew
                    break  # Found a match!
            if buddy:
                print 'Merging records with P(match) = ', prob
                print '   x2: ', get_weak_pairwise_features(currentrecord, rnew)
                currentrecord.display(indent='  ')
                print '   ----'
                buddy.display(indent='  ')
                currentrecord.merge(buddy)
                I.add(currentrecord)
                Inew.discard(buddy)
            else:
                Inew.add(currentrecord)
        return Inew


def merge_duped_records(tomerge):
    """
    Merges any entities containing the same record identifier.
    This is necessary due to placing records into multiple blocks.
    :param tomerge: Dictionary of [integer index, merged record]
    :return: entities: list of the final entities (record objects)
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
    entities = list()  # final output of merged records
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
            entities.append(merged)
        counter += 1
    return entities


def fast_strong_cluster(database):
    """
    Merges records with any strong features in common. Used as surrogate ground truth in CHT application
    Equivalent to running Swoosh using only strong matches, except much faster because of explicit graph exploration
    :param database: Database object
    :return labels: Cluster labels in the form of a dictionary [identifier, cluster_label]
    """
    print 'Clustering', len(database.records), 'records based on strong features...'
    strong2index = dict()  # [swoosh index, set of ad indices]  node --> edges
    graph = networkx.Graph()
    for _, record in database.records.iteritems():
        print '     Building strong feature --> index from record', record.line_indices
        indices = record.line_indices
        strong_features = record.get_features('strong')
        if not strong_features:  # no strong features, insert singular entity
            graph.add_edges_from(combinations_with_replacement(indices, 2))
        for strong in strong_features:
            if strong in strong2index:
                strong2index[strong].extend(list(indices))
            else:
                strong2index[strong] = list(indices)
    print '     ... finished.'
    for strong, indices in strong2index.iteritems():
        print 'Adding edges from indices', indices
        index0 = indices[0]
        for index in indices:
            graph.add_edge(index0, index)
    print 'Finding connected components...'
    connected_components = networkx.connected_components(graph)
    labels = dict()
    for cluster_label, component in enumerate(connected_components):
        for record_id in component:
            labels[record_id] = cluster_label
    print 'Finished. Found', cluster_label, 'connected components'
    return labels


def weak_connected_components(database, match_function, blocking_scheme):
    """
    Match and merge function operating on sets of records is equivalent to connected components.
    Direct implementation of this graphical approach.
    :param database: Database object to run ER on
    :param match_function: The trained match function
    :param blocking_scheme: BlockingScheme object
    :return identifier_to_cluster: Predicted labels, of dictionary form [record id, cluster label]:
    """
    print 'Finding weakly connected components'
    pairs = set()
    #for block_name, records in blocking_scheme.strong_blocks.iteritems():
    #    for pair in permutations(records, 2):
    #        pairs.add(pair)
    for counter, (block_name, records) in enumerate(blocking_scheme.weak_blocks.iteritems()):
        print '     Adding pairs from block', counter, 'of', len(blocking_scheme.weak_blocks)
        for pair in permutations(records, 2):
            pairs.add(pair)
    sparse_adjacency_matrix = list()  # symmetric, only stores lower triangular portion (i.e. pair(0)<pair(1))
    print 'Total number of pairs in blocking scheme to evaluate:', len(pairs)
    for counter, pair in enumerate(pairs):
        print '     Evaluating weak pair', counter, 'of', len(pairs)
        if pair[0] < pair[1]:
            r1 = database.records[pair[0]]
            r2 = database.records[pair[1]]
            if match_function.match(r1, r2):
                sparse_adjacency_matrix.append(pair)
    # Add singleton entities
    print 'Found', len(sparse_adjacency_matrix), 'weak matches.'
    for record_id, _ in database.records.iteritems():
        pair = (record_id, record_id)
        sparse_adjacency_matrix.append(pair)
    print 'Creating graph...'
    graph = networkx.Graph(sparse_adjacency_matrix)
    print 'Finding connected components...'
    connected_components = networkx.connected_components(graph)
    identifier_to_cluster = dict()
    for cluster_label, component in enumerate(connected_components):
        for record_id in component:
            identifier_to_cluster[record_id] = cluster_label
    print 'Finished weak connected components'
    return identifier_to_cluster
