""""
Randomly samples entities and dumps the records
"""
__author__ = 'mbarnes1'
import cProfile
import numpy as np
from random import shuffle


def main():
    # User parameters. Input labels are a csv of form annotation_index (0 indexed), text_index (1 indexed), entity_id
    number_samples = 1000
    truth_path = '../../entity_resolution_inputs/rebuild_phone_clusters.csv'
    out_path = '../../entity_resolution_inputs/rebuild_clusters_1000.csv'
    max_records_per_entity = 1000  # do not sample extremely large clusters (a hack to prevent entire subsample being a single large entity)
    max_entity_percentage = 0.3  # do not sample clusters which would constitute larger than this percentage of requested number of samples

    # Load the data
    ins = open(truth_path, 'r')
    entity_to_records = dict()
    text_to_annotation = dict()
    text_to_entity = dict()
    for counter, line in enumerate(ins):
        print 'Loading truth file line number', counter
        line = line.rstrip('\n').split(',')
        annotation_index = int(line[0])
        text_index = int(line[1])
        entity_id = int(line[2])
        text_to_annotation[text_index] = annotation_index
        text_to_entity[text_index] = entity_id
        if entity_id in entity_to_records:
            entity_to_records[entity_id].append(text_index)
        else:
            entity_to_records[entity_id] = [text_index]
    ins.close()

    # Sample the entities
    print 'Sampling entities'
    entities = entity_to_records.keys()
    entity_samples = np.random.choice(entities, min(len(entities), number_samples*2), replace=False)

    # Output the results
    counter = 0
    sampled_records = []
    while len(sampled_records) < number_samples:
        entity = entity_samples[counter]
        counter += 1
        records = entity_to_records[entity]
        if len(records) < min(max_records_per_entity, max_entity_percentage*number_samples):  # only clusters less than 10000 and less than 30%
            sampled_records.extend(records)
    print 'Shuffling results'
    shuffle(sampled_records)
    print 'Writing results'
    ins = open(out_path, 'w')
    ins.write('annotations_line (0 indexed), text_line (1 indexed), cluster_id')
    for sampled_record in sampled_records[:number_samples]:
        ins.write(str(text_to_annotation[sampled_record]) + ',' + str(sampled_record) + ',' + str(text_to_entity[sampled_record]) + '\n')
    ins.close()


if __name__ == '__main__':
    cProfile.run('main()')
