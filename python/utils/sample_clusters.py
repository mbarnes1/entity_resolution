""""
Randomly samples entities and dumps the records
"""
__author__ = 'mbarnes1'
import cProfile
import numpy as np
from random import shuffle


def main():
    # User parameters. Input labels are a csv of form annotation_index (0 indexed), text_index (1 indexed), entity_id
    number_samples = [10000, 100000, 1000000]
    number_databases = 2
    truth_path = '../../../entity_resolution_inputs/rebuild_phone_clusters.csv'
    out_path_prefix = '../../../entity_resolution_inputs/rebuild_clusters'
    max_records_per_entity = 10000  # do not sample extremely large clusters (a hack to prevent entire subsample being a single large entity)
    max_entity_percentage = 0.2  # do not sample clusters which would constitute larger than this percentage of requested number of samples

    # Load the data
    ins = open(truth_path, 'r')
    ins.next()  # skip header
    entity_to_text_index = dict()
    text_to_annotation = dict()
    text_to_entity = dict()
    for counter, line in enumerate(ins):
        print 'Loading truth file line number', counter
        line = line.rstrip('\n').split(',')
        try:
            annotation_index = int(line[0])
            text_index = int(line[1])
            entity_id = int(line[2])
            text_to_annotation[text_index] = annotation_index
            text_to_entity[text_index] = entity_id
            if entity_id in entity_to_text_index:
                entity_to_text_index[entity_id].append(text_index)
            else:
                entity_to_text_index[entity_id] = [text_index]
        except:
            print 'Unable to parse:', str(line)
    ins.close()

    # Sample the entities
    print 'Sampling entities'
    entities = entity_to_text_index.keys()
    shuffle(entities)

    # Output the results
    counter = 0
    for n in number_samples:
        for j in range(0, number_databases):
            out_path = out_path_prefix + str(j) + '_'+str(n)+'.csv'
            ins = open(out_path, 'w')
            ins.write('annotations_line (0 indexed), text_line (1 indexed), cluster_id\n')
            sampled_text_indices = []
            while len(sampled_text_indices) < n:
                entity = entities[counter]
                counter += 1
                text_indices = entity_to_text_index[entity]
                if len(text_indices) < min(max_records_per_entity, max_entity_percentage*number_samples):
                    sampled_text_indices.extend(text_indices)
            print 'Shuffling results'
            shuffle(sampled_text_indices)
            print 'Writing results'
            ins = open(out_path, 'w')
            ins.write('annotations_line (0 indexed), text_line (1 indexed), cluster_id\n')
            for sampled_text_index in sampled_text_indices[:number_samples]:
                ins.write(str(text_to_annotation[sampled_text_index]) + ',' + str(sampled_text_index) + ',' + str(text_to_entity[sampled_text_index]) + '\n')
            ins.close()


if __name__ == '__main__':
    cProfile.run('main()')
