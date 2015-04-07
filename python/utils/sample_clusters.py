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
    number_samples = [500, 1000]
    number_databases = 3
    cluster_path = '../test/test_annotations_10000_cleaned_strong_clusters.csv'  # '/home/scratch/trafficjam/entity_resolution_outputs/strong_clusters.csv'
    annotations_path = '../test/test_annotations_10000_cleaned.csv'  # '/home/scratch/trafficjam/deduped/US_Canada_Extraction_dedup.csv'
    header_path = '../test/test_annotations_10000_cleaned_header.csv'  # '/home/scratch/trafficjam/entity_resolution_outputs/US_Canada_Extraction_dedup_header.csv'

    database = Database(annotations_path, header_path=header_path)
    ins = open(cluster_path, 'r')
    cluster_to_indices = dict()
    next(ins)  # skip header
    for counter, line in enumerate(ins):
        print 'Loading cluster file', counter
        line = line.rstrip('\n').split(',')
        index = int(line[0])
        cluster_id = int(line[2])
        if cluster_id in cluster_to_indices:
            cluster_to_indices[cluster_id].append(index)
        else:
            cluster_to_indices[cluster_id] = [index]
    cluster_probabilities = list()
    for cluster, indices in cluster_to_indices.iteritems():
        cluster_probabilities.append(float(len(indices))/len(database.records))
    print 'Sampling clusters'
    cluster_samples = np.random.choice(cluster_to_indices.keys(), sum(number_samples)*number_databases, p=cluster_probabilities)
    counter = 0
    for n in number_samples:
        for j in range(0, number_databases):
            out_path = 'cluster_subsample'+str(j)+'_'+str(n)+'.csv'  # '/home/scratch/trafficjam/entity_resolution_outputs/cluster_subsample.csv'
            new_database = Database()
            new_database.feature_descriptor = database.feature_descriptor
            while len(new_database.records) < n:
                cluster = cluster_samples[counter]
                counter += 1
                indices = cluster_to_indices[cluster]
                for index in indices:
                    new_database.records[index] = database.records[index]
            new_database.dump(out_path)


if __name__ == '__main__':
    cProfile.run('main()')
