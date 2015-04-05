"""
Loads feature file, performs strong clustering, and dumps output to format:
line_index (ascending), poster_id (feature #1), cluster_id
"""
__author__ = 'mbarnes1'
from entityresolution import fast_strong_cluster, fast_strong_cluster_old
from database import Database
from metrics import Metrics
import timeit
import cProfile


def main():
    path = '../test/test_annotations_10000_cleaned.csv'
    out = open('../strong_clusters.csv', 'w')
    out.write('line_index, poster_id, cluster_id\n')
    database = Database(path)
    strong_clusters = fast_strong_cluster(database)
    strong_clusters_old = fast_strong_cluster_old(database)
    metrics = Metrics(strong_clusters_old, strong_clusters)
    metrics.display()
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
