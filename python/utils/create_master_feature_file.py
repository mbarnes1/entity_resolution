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
    for line1, line2 in izip(in1, in2):
        out.write(line1 + ',' + line2)
    out.close()

if __name__ == '__main__':
    cProfile.run('main()')
