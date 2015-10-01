__author__ = 'mbarnes1'
import cProfile
import numpy as np
#TODO: Output the cluster labels
#TODO: Loop through nrecords and p_corruption


def main():
    nrecords_list = [10000, 100000, 1000000]
    p_corruption_list = [0.1, 0.3, 0.5, 0.7]
    nfeatures = 100
    example_labels_path = '../../../entity_resolution_inputs/rebuild_clusters0_1000000.csv'  # file to model cluster sizes after

    print 'Loading example clusters'
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

    for nrecords in nrecords_list:
        for p_corruption in p_corruption_list:
            out_path = '../../../entity_resolution_inputs/synthetic_' + str(nrecords)+'_corrupt'+str(p_corruption*10)

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
            ins = open(out_path+".csv", 'w')
            ins_labels = open(out_path+"_labels.csv", 'w')
            ins_labels.write("line_to_use (0 indexed),Null,cluster_id\n")
            print 'Writing synthetic data'
            for cluster, size in cluster_sizes.iteritems():
                print 'Writing cluster of size', size
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
                    ins_labels.write(str(counter)+',,'+str(cluster)+'\n')
                    counter += 1
            ins.close()
            ins_labels.close()


if __name__ == '__main__':
    cProfile.run('main()')
