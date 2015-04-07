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
    number_samples = 1000
    cluster_path = '../test/test_annotations_10000_cleaned_strong_clusters.csv'  # '/home/scratch/trafficjam/entity_resolution_outputs/strong_clusters', 'w')
    annotations_path = '../test/test_annotations_10000_cleaned.csv'  # '/home/scratch/trafficjam/deduped/US_Canada_Extraction_dedup.csv'
    header_path = '../test/test_annotations_10000_cleaned_header.csv'  # 'US_Canada_Extraction_dedup_header.csv'
    out_path = 'cluster_subsample_'+str(number_samples)+'.csv'  # '/home/scratch/trafficjam/entity_resolution_outputs/cluster_subsample_'
    database = Database(annotations_path, header_path=header_path)
    ins = open(cluster_path, 'r')
    cluster_to_indices = dict()
    next(ins)  # skip header
    for line in ins:
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
    cluster_samples = np.random.choice(cluster_to_indices.keys(), number_samples, p=cluster_probabilities)
    new_database = Database()
    new_database.feature_descriptor = database.feature_descriptor
    counter = 0
    while len(new_database.records) < number_samples:
        cluster = cluster_samples[counter]
        counter += 1
        indices = cluster_to_indices[cluster]
        for index in indices:
            new_database.records[index] = database.records[index]
    new_database.dump(out_path)


if __name__ == '__main__':
    cProfile.run('main()')
