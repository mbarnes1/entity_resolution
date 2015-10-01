__author__ = 'mbarnes1'
import cProfile
import numpy as np


def main():
    nrecords = 10000
    example_labels_path = '../../../entity_resolution_inputs/rebuild_clusters1_10000.csv'  # file to model cluster sizes after
    out_path = '../../../entity_resolution_inputs/synthetic0_' + str(nrecords)
    nfeatures = 100
    p_corruption = 0.1

    print 'Loading loading example clusters'
    ins = open(example_labels_path, 'r')
    ins.next()  # skip header
    example_cluster_sizes = dict()
    for line in ins:
        line = line.rstrip('\n').split(',')
        cluster_id = int(line[2])
        if cluster_id in example_cluster_sizes:
            example_cluster_sizes[cluster_id] += 1
        else:
            example_cluster_sizes[cluster_id] = 1
    ins.close()
    cluster_sizes = dict()
    counter = 0
    for cluster, size in example_cluster_sizes.iteritems():
        to_add = min(size, nrecords-counter)
        cluster_sizes[counter] = to_add
        counter += to_add
        if counter >= nrecords:
            break
    nclusters = len(example_cluster_sizes)
    counter = 0
    ins = open(out_path, 'w')
    print 'Writing synthetic data'
    for cluster, size in cluster_sizes.iteritems():
        # Generate some synthetic data for this cluster
        true_features = cluster*np.ones((size, nfeatures))
        noise_indices = np.random.binomial(1, p_corruption, (size, nfeatures)).astype('bool')
        noise_features = np.random.randint(0, nclusters, (size, nfeatures))
        noise = np.multiply(noise_indices, noise_features)
        features = np.multiply(~noise_indices, true_features)
        synthetic = noise.astype(int) + features.astype(int)
        synthetic = synthetic.tolist()
        for row in synthetic:
            to_write = []
            for feature_counter, feature in enumerate(row):
                feature = 'f'+str(feature_counter)+'_'+str(feature)
                to_write.append(feature)
            line = ','.join(to_write)+'\n'
            ins.write(line)
    ins.close()



if __name__ == '__main__':
    cProfile.run('main()')
