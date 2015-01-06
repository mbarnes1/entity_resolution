"""
Implementation of many Entity Resolution evaluation metrics.
"""
import math
import sklearn.metrics
import numpy as np

__author__ = 'mbarnes1'


class Metrics(object):
    def __init__(self, labels_true, labels_pred):
        """
        :param labels_pred: Predicted cluster label of each sample. Dictionary of form [ad id, cluster label]
        :param labels_true: True cluster label of each sample. Dictionary of form [ad id, cluster label].
        Note: these are cluster labels, not class labels. The values do not have to correspond.
        """
        print 'Evaluating metrics...'
        self._n = float(len(labels_pred))
        self._labels_true = labels_true
        self._labels_pred = labels_pred
        self._clusters_pred = _cluster(labels_pred)
        self._clusters_true = _cluster(labels_true)
        self.number_entities = len(self._clusters_pred)
        self.pairwise_precision, self.pairwise_recall, self.pairwise_f1 = self._pairwise_precision_recall_f1()
        self.cluster_precision, self.cluster_recall, self.cluster_f1 = self._cluster_precision_recall_f1()
        self.closest_cluster_precision, self.closest_cluster_recall, self.closest_cluster_f1 = \
            self._closest_cluster_precision_recall_f1()
        self.aap, self.acp, self.k = self._average_author_cluster_purity()
        self.homogeneity, self.completeness, self.vmeasure = self._homogeneity_completeness_vmeasure()
        self.variation_of_information = self._variation_of_information()
        self.purity = self._purity()
        self._entity_sizes_pred = _get_entity_sizes(self._clusters_pred)
        self._entity_sizes_true = _get_entity_sizes(self._clusters_true)
        print 'metrics evaluated.'

    def _pairwise_precision_recall_f1(self):
        """
        Pairwise precision, pairwise recall, pairwise F1
        :return pairwise_precision: Pairwise precision, a float [0, 1]
        :return pairwise_recall: Pairwise recall, a float [0, 1]
        :return pairwise_f1: Pairwise F1, a float [0, 1]
        """
        pairwise_intersection_size = float(_intersection_size(self._clusters_pred, self._clusters_true))
        pairwise_precision = pairwise_intersection_size / _number_pairs(self._clusters_pred) if \
            _number_pairs(self._clusters_pred) else 1.0
        pairwise_recall = pairwise_intersection_size / _number_pairs(self._clusters_true) if \
            _number_pairs(self._clusters_true) else 0.0
        beta = 1.0
        pairwise_f1 = (1.0 + beta ** 2.0) * pairwise_precision * pairwise_recall / (
            (beta ** 2) * pairwise_precision + pairwise_recall) if \
            pairwise_precision or pairwise_recall else 0.0
        return pairwise_precision, pairwise_recall, pairwise_f1

    def _cluster_precision_recall_f1(self):
        """
        Cluster precision, cluster recall, cluster F1
        :return cluster_precision: Cluster precision, a float [0, 1]
        :return cluster_recall: Cluster recall, a float [0, 1]
        :return cluster_f1: Cluster F1, a float [0, 1]
        """
        cluster_precision = float(len(self._clusters_pred & self._clusters_true)) / len(self._clusters_pred)
        cluster_recall = float(len(self._clusters_pred & self._clusters_true)) / len(self._clusters_true)
        beta = 1
        cluster_f1 = (1.0 + beta ** 2.0) * cluster_precision * cluster_recall / (
            (beta ** 2.0) * cluster_precision + cluster_recall) if \
            cluster_precision or cluster_recall else 0.0
        return cluster_precision, cluster_recall, cluster_f1

    def _closest_cluster_precision_recall_f1(self):
        """
        Closest cluster precision, closest cluster recall, closest cluster F1
        :return closest_cluster_precision: Closest cluster precision, a float [0, 1]
        :return closest_cluster_recall: Closest cluster recall, a float [0, 1]
        :return closest_cluster_f1: Closest cluster F1, a float [0, 1]
        """
        closest_cluster_precision = _get_closest_cluster_precision_recall(self._clusters_pred, self._clusters_true)
        closest_cluster_recall = _get_closest_cluster_precision_recall(self._clusters_true, self._clusters_pred)
        beta = 1
        closest_cluster_f1 = ((1 + beta ** 2) * closest_cluster_precision * closest_cluster_recall /
                              ((beta ** 2) * closest_cluster_precision + closest_cluster_recall)) if \
            closest_cluster_precision or closest_cluster_recall else 0
        return closest_cluster_precision, closest_cluster_recall, closest_cluster_f1

    def _average_author_cluster_purity(self):
        """
        Average author purity, average cluster purity, and k
        :return average_author_purity:
        :return average_cluster_purity:
        :return k: Geometric mean of AAP and ACP
        """
        average_author_purity = _get_average_purity(self._clusters_true, self._clusters_pred)
        average_cluster_purity = _get_average_purity(self._clusters_pred, self._clusters_true)
        k = (average_author_purity * average_cluster_purity) ** 0.5
        return average_author_purity, average_cluster_purity, k

    def _homogeneity_completeness_vmeasure(self, beta=1):
        """
        :param beta: Recall is weighed more importantly if beta>1 and precision is weighed more importantly if beta<1
        :return homogeneity:
        :return completeness:
        :return vmeasure: Weighting of homogeneity and completeness. Harmonic mean when beta=1
        """
        linear_true, linear_pred = _linearize(self._labels_true, self._labels_pred)
        homogeneity, completeness, vmeasure = sklearn.metrics.homogeneity_completeness_v_measure(linear_true,
                                                                                                 linear_pred)
        return homogeneity, completeness, vmeasure

    def global_merge_distance(self, fs, fm, **kwargs):
        """
        The Global Merge Distance
        [Menestrain et al. 2010]
        :param fs: Split cost function handle
        :param fm: Merge cost function handle
        :param R: (optional) The predicted clusters. Defaults to initial clusters. Available to compute alternative GMDs
        :param S: (optional) The true clusters. Defaults to initial clusters. Available to compute alternative GMDs
        :return cost: The GMD
        """
        R = kwargs.get('R', self._clusters_pred)
        S = kwargs.get('S', self._clusters_true)
        # #
        M = dict()
        Rsizes = dict()
        for i, Ri in enumerate(R):
            for r in Ri:
                M[r] = i
            Rsizes[i] = len(Ri)
        # #
        cost = 0
        for i, Si in enumerate(S):
            pMap = dict()
            for r in Si:
                if M[r] not in pMap:
                    pMap[M[r]] = 0
                pMap[M[r]] += 1
            SiCost = 0
            totalRecs = 0
            for i, count in pMap.iteritems():
                if Rsizes[i] > count:
                    SiCost += fs(count, Rsizes[i] - count)
                Rsizes[i] += -count
                if totalRecs != 0:
                    SiCost += fm(count, totalRecs)
                totalRecs += count
            cost += SiCost
        return float(cost)

    def _variation_of_information(self):
        """
        :return vi: The variation of information
        """
        vi = (self._entropy(self._clusters_pred) + self._entropy(self._clusters_true) -
              2*self._mutual_information(self._clusters_pred, self._clusters_true))
        return vi

    def _entropy(self, clusters):
        """
        Determines the entropy H(clusters)
        :param clusters: Frozen set of clusters
        :return entropy:
        """
        entropy = 0
        for r in clusters:
            entropy -= len(r) / self._n * math.log(len(r) / self._n)
        return entropy

    def _mutual_information(self, R, S):
        """
        :param R: Frozen set of clusters
        :param S: Frozen set of clusters
        :return mi: Mutual information
        """
        mi = 0
        for r in R:
            for s in S:
                mi += float(len(r & s)) / self._n * math.log(len(r & s) * self._n / (len(r) * len(s))) \
                    if len(r & s) else 0
        return mi

    def _purity(self):
        """
        Purity, as defined at
        http://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
        :return total: The purity
        """
        total = 0
        for r in self._clusters_pred:
            subtotal = 0
            for s in self._clusters_true:
                subtotal = max(subtotal, len(r & s))
            total += subtotal
        total = float(total)/self._n
        return total

    def display(self):
        """
        Make weak feature classifier ROC curve, strong entity sizes, entity sizes, and block sizes figures
        Tries to catch error for when running on server, but it does not work
        """
        try:
            import matplotlib.pyplot as plt
            make_plots = True
        except ImportError or RuntimeError:
            make_plots = False
        if make_plots:
            ## True Entity Sizes
            plt.plot(self._entity_sizes_true[:, 0], self._entity_sizes_true[:, 1], 'ro')
            plt.xlabel('Number of ads in entity')
            plt.ylabel('Occurences')
            plt.title('True Entity Sizes')
            plt.show()

            ## Predicted Entity Sizes
            plt.plot(self._entity_sizes_pred[:, 0], self._entity_sizes_pred[:, 1], 'ro')
            plt.xlabel('Number of ads in entity')
            plt.ylabel('Occurences')
            plt.title('Predicted Entity Sizes')
            plt.show()


def _linearize(labels_a, labels_b):
    """
    :param labels_a: Labels of dictionary form [ad id, cluster label]
    :param labels_b: Labels of dictionary form [ad id, cluster label]
    :return linear_a: List of cluster labels from labels_a, with list indices matched to linear_b
    :return linear_b: List of cluster labels from labels_b, with list indices matched to linear_a
    """
    linear_a = list()
    linear_b = list()
    for id, cluster in labels_a.iteritems():
        linear_a.append(cluster)
        linear_b.append(labels_b[id])
    return linear_a, linear_b


def _cluster(labels):
    """
    :param labels: Cluster labels of dictionary form [record id, cluster id]
    :return clusters: Frozen set of clusters. Each cluster is a frozen set of record ids
    """
    cluster_to_record = dict()
    for id, cluster in labels.iteritems():
        if cluster in cluster_to_record:
            cluster_to_record[cluster].append(id)
        else:
            cluster_to_record[cluster] = [id]
    ## Convert to frozen sets
    clusters = set()
    for _, ads in cluster_to_record.iteritems():
        clusters.add(frozenset(ads))
    return frozenset(clusters)

# def _cluster(labels):
#     """
#     :param labels: Cluster labels. Iterable with length n.
#     :return clusters: Frozen set of clusters. Each cluster is a frozen set of ad ids
#     """
#     cluster_to_ads = dict()
#     for index, label in enumerate(labels):
#         if label in cluster_to_ads:
#             cluster_to_ads[label].append(index)
#         else:
#             cluster_to_ads[label] = [index]
#     ## Convert to frozen sets
#     clusters = set()
#     for _, ads in cluster_to_ads.iteritems():
#         clusters.add(frozenset(ads))
#     return frozenset(clusters)


def _intersection_size(clusters_estimate, clusters_truth):
    """
    Calculates the number of sample pairs in both clustering.
    Combinations, so order does not matter.
    :param clusters_estimate: Frozen set of estimated clusters. Each cluster is a frozen set of ad ids
    :param clusters_truth: Frozen set of true clusters. Each cluster is a frozen set of ad ids
    :return intersection_size: Number of sample pairs in both clusterings. Int.
    """
    edges_estimate = dict()
    for ads in clusters_estimate:
        for ad in ads:
            edges_estimate[ad] = ads
    edges_truth = dict()
    for ads in clusters_truth:
        for ad in ads:
            edges_truth[ad] = ads
    intersection_size = 0
    for ad, linked_ads in edges_estimate.iteritems():
        intersection_size += len(linked_ads & edges_truth[ad]) - 1  # subtract one for self-edge
    return intersection_size / 2


def _number_pairs(clusters):
    """
    Calculates the number of intra-cluster pairs.
    Combinations, so order does not matter
    :param clusters: Frozen set of clusters. Each cluster is a frozen set of ad ids
    :return number_pairs: Number of pairs, int.
    """
    number_pairs = 0
    for ads in clusters:
        number_pairs += len(ads) * (len(ads) - 1) / 2
    return number_pairs


def _get_closest_cluster_precision_recall(clusters_r, clusters_s):
    """
    Closest cluster precision or recall, depending on order of inputs
    Precision = get_closest_cluster_precision_recall(clusters_predicted, cluster_truth)
    Recall = get_closest_cluster_precision_recall(clusters_predicted, clusters_truth)
    :param clusters_r: Frozen set of clusters.
    :param clusters_s: Frozen set of clusters
    :return metric: Precision or recall, depending on other of inputs
    """
    total_similarity = 0
    for r in clusters_r:
        max_similarity = 0
        for s in clusters_s:
            max_similarity = max(max_similarity, _jaccard(r, s))
        total_similarity += max_similarity
    return total_similarity / len(clusters_r)


def _jaccard(r, s):
    """
    Jaccard similarity index
    :param r: Set
    :param s: Set
    :return j: jacccard coefficient
    """
    j = float(len(r & s)) / len(r | s)
    return j


def _get_average_purity(clusters_r, clusters_s):
    """
    Average author purity or average cluster purity, depending on order of inputs
    AAP = get_average_purity(clusters_true, clusters_pred)
    ACP = get_average_purity(clusters_pred, clusters_true)
    :param clusters_r: Frozen set of clusters
    :param clusters_s: Frozen set of clusters
    :return:
    """
    n = 0
    total = 0
    for r in clusters_r:
        n += len(r)
        for s in clusters_s:
            total += float(len(r & s) ** 2) / len(r)
    return float(total) / n


def _get_entity_sizes(clusters):
    """
    Determines the entity sizes, for plotting
    :param clusters: Frozen set of clusters, each a frozen set of ad ids
    :return xy: number_bins x 2 array of entity sizes. First column is sorted size, second column is number of occurences
    """
    size_occurences = dict()
    for cluster in clusters:
        size = len(cluster)
        if size in size_occurences:
            size_occurences[size] += 1
        else:
            size_occurences[size] = 1
    x = np.zeros(len(size_occurences),)
    y = np.zeros(len(size_occurences),)
    for counter, (size, occurences) in enumerate(size_occurences.iteritems()):
        x[counter] = size
        y[counter] = occurences
    xy = np.vstack((x, y)).T
    xy = xy[xy[:, 0].argsort()]
    return xy