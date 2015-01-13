__author__ = 'mbarnes1'
import numpy as np
from itertools import izip
from sklearn.metrics import auc


class RocCurve(object):
    def __init__(self, labels, prob):
        print 'Calculating TPR & FPR...'
        ind = np.argsort(-prob)  # sort in descending order
        labels = labels.astype('bool')
        self.labels = labels[ind]
        self.prob = prob[ind]
        self.tpr = np.cumsum(self.labels).astype('float')/np.sum(self.labels)
        self.fpr = np.cumsum(~self.labels).astype('float')/np.sum(~self.labels)
        self.auc = auc(self.fpr, self.tpr)
        print 'Area Under Curve:', self.auc
        print '     Threshold, FPR, TPR'
        for prob, fpr, tpr in izip(self.prob, self.fpr, self.tpr):
            print "{:10.4f}, {:10.4f}, {:10.4f}".format(prob, fpr, tpr)

    # Write tpr and fpr to a flat file
    def write(self, path):
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