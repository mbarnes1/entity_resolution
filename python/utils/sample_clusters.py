"""
Randomly samples entities and dumps the records
"""
__author__ = 'mbarnes1'
import cProfile
import numpy as np
from random import shuffle


def main():
    # User parameters. Input labels are a csv of form record_id, entity_id
    number_samples = 10
    truth_path = '../test/labels.csv'
    out_path = '../test/samples.csv'
    max_records_per_entity = 1000  # do not sample extremely large clusters (a hack to prevent entire subsample being a single large entity)
    max_entity_percentage = 0.3  # do not sample clusters which would constitute larger than this percentage of requested number of samples

    # Load the data
    ins = open(truth_path, 'r')
    entity_to_records = dict()
    for counter, line in enumerate(ins):
        print 'Loading truth file line number', counter
        line = line.rstrip('\n').split(',')
        record_id = int(line[0])
        entity_id = int(line[1])
        if entity_id in entity_to_records:
            entity_to_records[entity_id].append(record_id)
        else:
            entity_to_records[entity_id] = [record_id]
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
    shuffle(sampled_records)
    ins = open(out_path, 'w')
    for sampled_record in sampled_records[:number_samples]:
        ins.write(str(sampled_record)+'\n')
    ins.close()


if __name__ == '__main__':
    cProfile.run('main()')