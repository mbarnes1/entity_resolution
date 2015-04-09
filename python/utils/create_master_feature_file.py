"""
Adds header to LM50 features, only keeps first 30 (last 20 are from trafficking records)
"""
__author__ = 'mbarnes1'
from itertools import izip
import cProfile


def main():
    LM_path = '/home/scratch/trafficjam/entity_resolution_inputs/US_Canada_LM50.csv'
    LM_header_path = '/home/scratch/trafficjam/entity_resolution_inputs/US_Canada_LM50_header.csv'
    annotations_path = '/home/scratch/trafficjam/deduped/US_Canada_Extraction_dedup.csv'
    annotations_header_path = '/home/scratch/trafficjam/entity_resolution_inputs/US_Canada_Extraction_dedup_header.csv'
    header_path = '/home/scratch/trafficjam/entity_resolution_inputs/master_header_all.csv'
    out_path = '/home/scratch/trafficjam/entity_resolution_inputs/master.csv'

    # Create the master header file
    out = open(header_path, 'w')
    in1 = open(annotations_header_path, 'r')
    in2 = open(LM_header_path, 'r')
    for counter, (line1, line2) in enumerate(izip(in1, in2)):
        line1 = line1.strip('\n')
        line2 = ','.join(line2.strip('\n').split(',')[0:30])
        new_line = line1 + ',' + line2 + '\n'
        if counter is 0:
            feature_names = new_line
        out.write(new_line)
    out.close()
    in1.close()
    in2.close()

    # Create the master feature file
    out = open(out_path, 'w')
    out.write(feature_names)
    in1 = open(annotations_path, 'r')
    in2 = open(LM_path, 'r')
    for counter, (line1, line2) in enumerate(izip(in1, in2)):
        print 'Writing features from line', counter
        line1 = line1.strip('\n')
        line2 = ','.join(line2.strip('\n').split(',')[0:30])
        out.write(line1 + ',' + line2 + '\n')
    out.close()
    in1.close()
    in2.close()

if __name__ == '__main__':
    cProfile.run('main()')
