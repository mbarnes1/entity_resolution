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
    annotations_path = '/home/scratch/trafficjam/entity_resolution_inputs/master.csv'
    header_path = '/home/scratch/trafficjam/entity_resolution_inputs/master_header_all.csv'
    database = Database(annotations_path, header_path=header_path)

    ins = open(cluster_path, 'r')
    cluster_to_indices = dict()
    next(ins)  # skip header
    for cluster_counter, line in enumerate(ins):
        print 'Loading cluster file', cluster_counter
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
    for n in number_samples:
        for j in range(0, number_databases):
            out_path = '/home/scratch/trafficjam/entity_resolution_inputs/subsample'+str(j)+'_'+str(n)+'.csv'
            new_database = Database()
            new_database.feature_descriptor = database.feature_descriptor
            while len(new_database.records) < n:
                cluster = cluster_samples[cluster_counter]
                cluster_counter += 1
                indices = cluster_to_indices[cluster]
                if len(indices) < min(10000, 0.05*n):  # only clusters less than 5000 or 5%
                    for index in indices:
                        new_database.records[index] = database.records[index]
            new_database.dump(out_path)


if __name__ == '__main__':
    cProfile.run('main()')
