__author__ = 'mbarnes1'
import numpy as np


class RocCurve(object):
    def __init__(self, labels, prob):
        ind = np.argsort(-prob)  # sort in descending order
        labels = labels.astype('bool')
        self.labels = labels[ind]
        self.prob = prob[ind]
        self.tpr = np.cumsum(self.labels).astype('float')/np.sum(self.labels)
        self.fpr = np.cumsum(~self.labels).astype('float')/np.sum(~self.labels)

    # Write tpr and fpr to a flat file
    def write(self, path):
        rates = np.column_stack((self.tpr, self.fpr))
        np.savetxt(path, rates, delimiter=",", header='tpr,fpr')

    def make_plot(self, black_scheme=True):
        import pylab as pl
        if black_scheme:
            from mpltools import style
            style.use('dark_background')
            tick_color = 'k'
        else:
            tick_color = 'w'
        fig = pl.figure(figsize=(14, 6))
        ax = fig.add_subplot(111)    # The big subplot
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        # Turn off axis lines and ticks of the big subplot

        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor=tick_color, top='off', bottom='off', left='off', right='off')

        c1 = pl.rcParams['axes.color_cycle'][2]
        c2 = pl.rcParams['axes.color_cycle'][1]

        tpr_l, tpr_u = wilson_confidence(self.tpr, np.sum(self.labels))
        fpr_l, fpr_u = wilson_confidence(self.fpr, np.sum(~self.labels))
        ax1.plot(self.fpr, self.tpr, color=c1)
        ax1.fill_between(fpr_l, 0, tpr_u, color=c1, alpha=0.2)
        ax1.fill_between(fpr_u, 0, tpr_l, color=tick_color)
        ax1.plot([0, 1], [0, 1], 'w--')
        #insertpt = np.searchsorted(self.prob, 0.5, sorter=range(len(self.prob)-1, -1, -1))
        #ax1.plot(self.fpr[insertpt], self.tpr[insertpt], color=c1, marker='o')

        ax2.fill_between(fpr_l, 0, tpr_u, color=c1, alpha=0.2)
        ax2.fill_between(fpr_u, 0, tpr_l, color=tick_color)
        p1, = ax2.semilogx(self.fpr, self.tpr, color=c1)
        #p2, = ax2.semilogx(self.fpr[insertpt], self.tpr[insertpt], color=c1, marker='o')
        #ax2.legend([p1, p2], ['Surrogate', '$P = 0.5$'], 'lower right')

        # Set common labels
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('P(strong | weak)', y=1.03, fontsize=18)
        pl.savefig('roc.pdf', format='pdf')
        pl.show()


def wilson_confidence(p, n):
    n = n.astype('float')
    # Returns the wilson confidence interval for all roc points
    z = 1.96  # for 95% confidence interval
    upper = 1/(1+1/n*pow(z, 2)) * (p + 1/(2*n)*pow(z, 2) + z*np.sqrt(1/n*p*(1-p)+1/(4*pow(n, 2))*pow(z, 2)))
    lower = 1/(1+1/n*pow(z, 2)) * (p + 1/(2*n)*pow(z, 2) - z*np.sqrt(1/n*p*(1-p)+1/(4*pow(n, 2))*pow(z, 2)))
    return lower, upper