__author__ = 'mbarnes1'
import numpy as np
from itertools import izip
from sklearn.metrics import auc
from bisect import bisect, bisect_left
from copy import copy


class RocCurve(object):
    def __init__(self, labels, prob):
        """
        :param labels: List of the true labels
        :param prob: List of the corresponding probabilities
        :return:
        """
        print 'Calculating TPR & FPR...'
        ind = np.argsort(-prob)  # sort in descending order
        labels = labels.astype('bool')
        self.labels = labels[ind]
        self.prob = prob[ind]
        self.class_balance = float(sum(self.labels))/len(labels)
        print '     Class balance of', self.class_balance

        self.tpr = np.cumsum(self.labels).astype('float')/np.sum(self.labels)
        self.recall = copy(self.tpr)  # TPR = recall, by definition
        self.fpr = np.cumsum(~self.labels).astype('float')/np.sum(~self.labels)
        self.precision = np.cumsum(self.labels).astype('float')/np.arange(1, len(self.labels)+1)
        self.f1 = 2*self.precision*self.recall/(self.precision + self.recall)

        precision_wilson_n = np.arange(1, len(self.labels)+1)
        recall_wilson_n = np.sum(self.labels)*np.ones(self.recall.size)

        # Reverse all the arrays and lists to prob is in ascending order
        self.labels = self.labels[::-1]
        self.prob = self.prob[::-1]
        self.tpr = self.prob[::-1]
        self.recall = self.recall[::-1]
        self.fpr = self.fpr[::-1]
        self.precision = self.precision[::-1]
        self.f1 = self.f1[::-1]
        precision_wilson_n = precision_wilson_n[::-1]
        recall_wilson_n = recall_wilson_n[::-1]

        # Pad with a value at prob=1.0
        self.prob = np.append(self.prob, 1.0)
        self.recall = np.append(self.recall, 0.0)
        self.precision = np.append(self.precision, 1.0)
        self.f1 = np.append(self.f1, 0.0)
        precision_wilson_n = np.append(precision_wilson_n, 1.0)  # theoretically zero, but confidence interval would go to infinity
        recall_wilson_n = np.append(recall_wilson_n, recall_wilson_n[-1])
        self.precision_lower_ci, self.precision_upper_ci = wilson_confidence(self.precision, precision_wilson_n)
        self.recall_lower_ci, self.recall_upper_ci = wilson_confidence(self.recall, recall_wilson_n)

        self.auc = auc(self.fpr, self.tpr)
        print '     Label, Threshold, Precision, Recall'
        for label, prob, prec, recall in izip(self.labels, self.prob, self.precision, self.recall):
            print "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(label, prob, prec, recall)
        print 'Area Under Curve:', self.auc
        print '     Threshold, FPR, TPR'
        for prob, fpr, tpr in izip(self.prob, self.fpr, self.tpr):
            print "{:10.4f}, {:10.4f}, {:10.4f}".format(prob, fpr, tpr)

    def get_recall(self, threshold):
        """
        Returns the recall at the specified threshold
        :param threshold:
        :return recall:
        :return recall_lower_ci: 95% lower confidence interval, based on Wilson score
        :return recall_upper_ci: 95% upper confidence interval, based on Wilson score
        """
        if 0.0 <= threshold <= 1.0:
            index = bisect_left(self.prob, min(1.0, threshold))  # first occurence of this threshold
            recall = self.recall[index]
            recall_lower_ci = self.recall_lower_ci[index]
            recall_upper_ci = self.recall_upper_ci[index]
        elif threshold > 1.0:
            recall = 0.0
            recall_lower_ci = 0.0
            recall_upper_ci = 0.0
        else:
            recall = 1.0
            recall_lower_ci = 1.0
            recall_upper_ci = 1.0
        print 'Validation set match recall', recall
        return recall, recall_lower_ci, recall_upper_ci

    def get_precision(self, threshold):
        """
        Returns the precision at the specified threshold
        :param threshold:
        :return precision:
        :return precision_lower_ci: Precision 95% lower bound
        :return precision_upper_ci: Precision 95% upper bound
        """
        if 0.0 <= threshold <= 1.0:
            index = bisect(self.prob, min(1.0, threshold)) - 1  # first occurence of this threshold
            if index >= 0:
                precision = self.precision[index]
                precision_upper_ci = self.precision_upper_ci[index]
                precision_lower_ci = self.precision_lower_ci[index]
            elif index == -1:
                precision = self.precision[0]
                precision_upper_ci = self.precision_upper_ci[0]
                precision_lower_ci = self.precision_lower_ci[0]
            else:
                raise IndexError
        elif threshold > 1.0:
            precision = 1.0
            precision_lower_ci = 1.0
            precision_upper_ci = 1.0
        else:
            precision = 0.0
            precision_lower_ci = 0.0
            precision_upper_ci = 0.0

        return precision, precision_lower_ci, precision_upper_ci

    def write(self, path):
        """
        Write TPR and FPR to a flat file
        :param path: File path to write
        """
        rates = np.column_stack((self.tpr, self.fpr))
        np.savetxt(path, rates, delimiter=",", header='tpr,fpr')

    def make_plot(self, title='P(strong | weak)'):
        import pylab as pl
        f, (ax1, ax2) = pl.subplots(1, 2, sharey=True, figsize=(14, 6), facecolor='white')
        tpr_l, tpr_u = wilson_confidence(self.tpr, np.sum(self.labels))
        fpr_l, fpr_u = wilson_confidence(self.fpr, np.sum(~self.labels))

        c1 = pl.rcParams['axes.color_cycle'][2]
        ax1.plot(self.fpr, self.tpr, color=c1)
        ax1.grid(True)
        ax1.fill_between(fpr_l, 0, tpr_u, color=c1, alpha=0.2)
        ax1.fill_between(fpr_u, 0, tpr_l, color='w')
        ax1.set_xlim([0, 1.0])
        ax1.set_ylim([0, 1.0])

        ax2.fill_between(fpr_l, 0, tpr_u, color=c1, alpha=0.2)
        ax2.fill_between(fpr_u, 0, tpr_l, color='w')
        ax2.semilogx(self.fpr, self.tpr, color=c1)
        ax2.set_xlim([0, 1.0])
        ax2.grid(True)

        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax2.set_xlabel('Log FPR')
        ax2.set_ylabel('True Positive Rate')
        pl.suptitle(title)

        pl.show()
        #pl.savefig('roc.png', facecolor='white', edgecolor='none')


def wilson_confidence(p, n):
    n = n.astype('float')
    # Returns the wilson confidence interval for all roc points
    z = 1.96  # for 95% confidence interval
    upper = 1/(1+1/n*pow(z, 2)) * (p + 1/(2*n)*pow(z, 2) + z*np.sqrt(1/n*p*(1-p)+1/(4*pow(n, 2))*pow(z, 2)))
    lower = 1/(1+1/n*pow(z, 2)) * (p + 1/(2*n)*pow(z, 2) - z*np.sqrt(1/n*p*(1-p)+1/(4*pow(n, 2))*pow(z, 2)))
    return lower, upper