"""
Loads feature file, performs strong clustering, and dumps output to format:
line_index (ascending), poster_id (feature #1), cluster_id
"""
__author__ = 'mbarnes1'
import sys
sys.path.append('..')
from entityresolution import fast_strong_cluster
from database import Database
import cProfile


def main():
    path = '/home/scratch/trafficjam/rebuild/annotations.csv'
    header_path = '/home/scratch/trafficjam/entity_resolution_inputs/rebuild_annotations_header.csv'
    out = open('/home/scratch/trafficjam/entity_resolution_inputs/rebuild_phone_clusters_faster.csv', 'w')
    out.write('annotations_line (0 indexed), text_line (1 indexed), cluster_id\n')
    database = Database(path, header_path=header_path)
    strong_clusters = fast_strong_cluster(database)
    line_indices = strong_clusters.keys()
    line_indices.sort()
    for line_index in line_indices:
        poster_id = database.records[line_index].features[0]
        if poster_id:
            (poster_id,) = poster_id  # unpack from set
        else:
            poster_id = ''
        cluster_id = strong_clusters[line_index]
        out.write(str(line_index)+','+str(poster_id)+','+str(cluster_id)+'\n')
    out.close()

if __name__ == '__main__':
    cProfile.run('main()')
