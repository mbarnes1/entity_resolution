"""
Randomly samples entire clusters and dumps their results
"""
__author__ = 'mbarnes1'
import sys
sys.path.append('..')
from database import Database
import cProfile
import numpy as np


def main():
    number_samples = [10000, 100000, 1000000]
    number_databases = 3
    cluster_path = '/home/scratch/trafficjam/entity_resolution_outputs/strong_clusters.csv'
    annotations_path = '/home/scratch/trafficjam/deduped/US_Canada_Extraction_dedup.csv'
    header_path = '/home/scratch/trafficjam/entity_resolution_outputs/US_Canada_Extraction_dedup_header.csv'

    ins = open(cluster_path, 'r')
    cluster_to_indices = dict()
    next(ins)  # skip header
    for counter, line in enumerate(ins):
        print 'Loading cluster file', counter
        line = line.rstrip('\n').split(',')
        index = int(line[0])
        poster_id = int(line[1])
        cluster_id = int(line[2])
        if cluster_id in cluster_to_indices:
            cluster_to_indices[cluster_id].append(index)
        else:
            cluster_to_indices[cluster_id] = [index]
    cluster_probabilities = list()
    for cluster, indices in cluster_to_indices.iteritems():
        cluster_probabilities.append(float(len(indices))/len(database.records))
    print 'Sampling clusters'
    cluster_samples = np.random.choice(cluster_to_indices.keys(), min(850000, sum(number_samples)*number_databases), p=cluster_probabilities)  # only ~850,000 strong clusters available
    cluster_counter = 0
    record_counter = 0
    for n in number_samples:
        for j in range(0, number_databases):
            out_path = '/home/scratch/trafficjam/entity_resolution_outputs/subsample_indices'+str(j)+'_'+str(n)+'.csv'
            outs = open(out_path, 'w')
            outs.write('line_index, poster_id, cluster_id\n')
            while record_counter < n:
                cluster = cluster_samples[cluster_counter]
                cluster_counter += 1
                indices = cluster_to_indices[cluster]
                if len(indices) < min(.05*n, 10000):  # don't use any clusters with more than 10,000 records or 5% of database size
                    for line_index in indices:
                        poster_id = database.records[line_index].features[0]
                        if poster_id:
                            (poster_id,) = poster_id  # unpack from set
                        else:
                            poster_id = ''
                        outs.write(str(line_index)+','+str(poster_id)+','+str(cluster)+'\n')
            outs.close()


if __name__ == '__main__':
    cProfile.run('main()')
