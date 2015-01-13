__author__ = 'mbarnes1'
import csv
import re


def main():
    file_input_1 = 'Abt.csv'
    file_input_2 = 'Buy.csv'
    f1 = open(file_input_1, 'r')
    id_feature_dict = dict()
    reader = csv.reader(f1)
    next(reader)
    for line in reader:
        identifier = line[0]
        print 'Loading record', identifier
        features_with_commas = line[1:3]
        features = []
        for feature in features_with_commas:
            features.append(re.sub(r',', '', feature))
        if line[3]:
            cost = line[3][1:]  # remove dollar sign
            cost = re.sub(r',', '', cost)  # remove commas
            features.append(cost)
        else:
            features.append('')
        features = ','.join(features)
        if identifier not in id_feature_dict:
            id_feature_dict[identifier] = features
        else:
            raise KeyError('Duplicate keys should not exist')
    print len(id_feature_dict), 'records in Abt'
    f1.close()
    f2 = open(file_input_2, 'r')
    reader = csv.reader(f2)
    next(reader)  # skip header line
    for line in reader:
        identifier = line[0]
        print 'Loading record', identifier
        features_with_commas = line[1:3]
        features = []
        for feature in features_with_commas:
            features.append(re.sub(r',', '', feature))
        if line[4]:  # skip manufacturer field, not in Abt
            cost = line[4][1:]  # remove dollar sign
            cost = re.sub(r',', '', cost)  # remove commas
            features.append(cost)
        else:
            features.append('')
        features = ','.join(features)
        if identifier not in id_feature_dict:
            id_feature_dict[identifier] = features
        else:
            raise KeyError('Duplicate keys should not exist')
    print len(id_feature_dict), 'records in total'
    f2.close()

    mapping = 'abt_buy_perfectMapping.csv'
    features_file = 'merged.csv'
    labels_file = 'labels.csv'

    f_mapping = open(mapping, 'r')
    f_mapping.next()
    f_features = open(features_file, 'w')
    f_features.write('name,description,price\n')
    f_features.write('string,string,float\n')
    f_features.write('weak,weak,weak\n')
    f_features.write('noblock,noblock,noblock\n')
    f_features.write('levenshtein,levenshtein,numerical_difference\n')
    f_labels = open(labels_file, 'w')
    id1_to_id2 = dict()
    id2_to_id1 = dict()
    for counter, line in enumerate(f_mapping):
        mapping = line.rstrip('\n').rstrip('\r').split(',')
        id_1 = mapping[0]
        id_2 = mapping[1]
        if id_1 not in id1_to_id2:
            id1_to_id2[id_1] = [id_2]
        else:
            id1_to_id2[id_1].append(id_2)
        if id_2 not in id2_to_id1:
            id2_to_id1[id_2] = [id_1]
        else:
            id2_to_id1[id_2].append(id_1)
    if len(id1_to_id2) + len(id2_to_id1) != len(id_feature_dict):
        raise Exception('Number of records does not match expected')
    clusters = fast_cluster(id1_to_id2, id2_to_id1)
    identifier_check = set()
    for counter, cluster in enumerate(clusters):
        for identifier in cluster:
            if identifier not in identifier_check:
                identifier_check.add(identifier)
            else:
                raise KeyError('Identifier ' + identifier + ' should only appear once')
            f_features.write(id_feature_dict[identifier])
            f_features.write('\n')
            f_labels.write(str(counter))
            f_labels.write('\n')
    f_mapping.close()
    f_features.close()
    f_labels.close()


def fast_cluster(id1_to_id2, id2_to_id1):
    """
    Finds clusters from dual hash tables
    :param hash1: dict [id1, list of id2's]
    :param hash2: dict [id2, list of id1's]
    :return clusters: list of clusters, where each cluster is a set of ids
    """
    # Determine connected components
    explored_id1 = set()
    explored_id2 = set()
    to_explore = list()
    clusters = []
    for id1, id2_list in id1_to_id2.iteritems():
        if id1 not in explored_id1:
            explored_id1.add(id1)
            cc = set()
            cc.add(id1)
            to_explore.extend(id2_list)
            while to_explore:
                id2 = to_explore.pop()
                if id2 not in explored_id2:
                    cc.add(id2)
                    explored_id2.add(id2)
                    connected_id1_list = id2_to_id1[id2]
                    for connected_id1 in connected_id1_list:
                        if connected_id1 not in explored_id1:
                            cc.add(connected_id1)
                            explored_id1.add(connected_id1)
                            to_explore.extend(id1_to_id2[connected_id1])

            # Found complete connected component, save labels
            clusters.append(cc)
    return clusters

if __name__ == '__main__':
    main()