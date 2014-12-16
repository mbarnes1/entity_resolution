__author__ = 'mbarnes1'
from database import Synthetic
from copy import deepcopy
from entityresolution import EntityResolution
from pairwise_features import generate_pair_seed
from itertools import izip
import numpy as np
from metrics import Metrics
import matplotlib.pyplot as plt
plt.ion()
from matplotlib.widgets import Slider
import cProfile
import matplotlib.cm as cm
from new_metrics import NewMetrics


class SyntheticExperiment(object):
    class ResultsPlot(object):
        """
        2D plot of all the entities and logistic regression decision boundaries (ellipsoids)
        """
        def __init__(self, experiment):
            """
            :param experiment: SyntheticExperiment parent object
            """
            self._experiment = experiment
            color_seed = cm.rainbow(np.linspace(0, 1, len(experiment.uncorrupted_synthetic_test.labels)))
            np.random.shuffle(color_seed)
            self._color_seed = color_seed  # np.random.rand(len(experiment._uncorrupted_synthetic_test.labels))

            self._corruption_index = 0
            self._threshold_index = 0

            # Plot the metrics
            self._figures = list()
            self._axes = list()

            # Pairwise F1
            self._figures.append(plt.figure())
            ax = self._figures[0].add_subplot(111)
            ax.grid(linestyle='--')
            self._axes.append([ax])
            pairwise_f1 = list()
            for threshold_index, threshold in enumerate(experiment.thresholds):
                pairwise_f1.append(experiment.metrics[self._corruption_index][threshold_index].pairwise_f1)
            self.pf1, = ax.plot(experiment.thresholds, pairwise_f1)
            self.pf1_dot, = ax.plot(experiment.thresholds[self._threshold_index], pairwise_f1[self._threshold_index],
                                     'bo', markersize=10, label='Operating Point')
            plt.legend(loc='upper left')
            self._axes[0][0].set_xlabel('Threshold')
            self._axes[0][0].set_ylabel('Pairwise F1')
            self._axes[0][0].set_title('Pairwise F1')
            self._axes[0][0].set_ylim([0, 1.0])

            # New metric
            self._figures.append(plt.figure())
            ax = self._figures[1].add_subplot(111)
            ax.grid(linestyle='--')
            self._axes.append([ax])
            new_metrics = list()
            for threshold_index, threshold in enumerate(experiment.thresholds):
                new_metrics.append(experiment.new_metrics[self._corruption_index][threshold_index].net_expected_cost)
            self.new_metrics, = ax.plot(experiment.thresholds, new_metrics)
            self.new_metrics_dot, = ax.plot(experiment.thresholds[self._threshold_index],
                                            new_metrics[self._threshold_index], 'bo', markersize=10,
                                            label='Operating Point')
            plt.legend(loc='upper left')
            self._axes[1][0].set_xlabel('Threshold')
            self._axes[1][0].set_ylabel('New Metric')
            self._axes[1][0].set_title('New Metric')


            # Plot the samples
            self._figures.append(plt.figure())
            ax0 = self._figures[-1].add_subplot(121, aspect='equal')
            ax1 = self._figures[-1].add_subplot(122, aspect='equal')
            self._axes.append([ax0, ax1])
            plt.subplots_adjust(bottom=0.25)  # changes location of plot's bottom left hand corner (no slider overlap)

            # plot ground truth
            true_labels = experiment.uncorrupted_synthetic_test.labels
            experiment.synthetic_test[self._corruption_index].plot(true_labels, title='True Clustering',
                                                                   color_seed=self._color_seed, ax=self._axes[-1][0])

            # plot predicted cluster
            predicted_labels = experiment.predicted_labels[self._corruption_index][self._threshold_index]
            experiment.synthetic_test[self._corruption_index].plot(predicted_labels, title='Predicted Clustering',
                                                                   color_seed=self._color_seed, ax=self._axes[-1][1])
            
            # make the sliders
            axframe = plt.axes([0.125, 0.1, 0.775, 0.03])
            self.sframe = Slider(axframe, 'Noise', 0, len(experiment.corruption_multipliers)-1, valinit=0, valfmt='%d')
            axframe2 = plt.axes([0.125, 0.15, 0.775, 0.03])
            self.sframe2 = Slider(axframe2, 'Threshold', 0, len(experiment.thresholds)-1, valinit=0, valfmt='%d')

            # connect callback to slider
            self.sframe.on_changed(self.update)
            self.sframe2.on_changed(self.update2)
            plt.show()

        # call back function
        def update(self, _):
            """
            Updating the corruption
            """
            corruption_index = int(np.floor(self.sframe.val))
            if corruption_index != self._corruption_index:
                true_labels = self._experiment.uncorrupted_synthetic_test.labels
                predicted_labels = self._experiment.predicted_labels[corruption_index][self._threshold_index]
                self._axes[-1][0].clear()
                self._axes[-1][1].clear()
                self._experiment.synthetic_test[corruption_index].plot(true_labels, title='True Clustering',
                                                                       color_seed=self._color_seed, ax=self._axes[-1][0])
                self._experiment.synthetic_test[corruption_index].plot(predicted_labels, title='Predicted Clustering',
                                                                       color_seed=self._color_seed, ax=self._axes[-1][1])
                pairwise_f1 = list()
                new_metrics = list()
                for threshold_index, threshold in enumerate(self._experiment.thresholds):
                    pairwise_f1.append(self._experiment.metrics[corruption_index][threshold_index].pairwise_f1)
                    new_metrics.append(self._experiment.new_metrics[corruption_index][threshold_index].net_expected_cost)
                self.pf1.set_ydata(pairwise_f1)
                self.pf1_dot.set_ydata(pairwise_f1[self._threshold_index])
                self.new_metrics.set_ydata(new_metrics)
                self.new_metrics_dot.set_ydata(new_metrics[self._threshold_index])
                self._figures[0].canvas.draw()
                self._figures[1].canvas.draw()
                self._corruption_index = corruption_index

        def update2(self, _):
            """
            Updating the threshold
            """
            corruption_index = self._corruption_index
            threshold_index = int(np.floor(self.sframe2.val))
            if threshold_index != self._threshold_index:
                true_labels = self._experiment.uncorrupted_synthetic_test.labels
                predicted_labels = self._experiment.predicted_labels[corruption_index][threshold_index]
                self._axes[-1][0].clear()
                self._axes[-1][1].clear()
                self._experiment.synthetic_test[corruption_index].plot(true_labels, title='True Clustering',
                                                                       color_seed=self._color_seed, ax=self._axes[-1][0])
                self._experiment.synthetic_test[corruption_index].plot(predicted_labels, title='Predicted Clustering',
                                                                       color_seed=self._color_seed, ax=self._axes[-1][1])
                self.pf1_dot.set_xdata(self._experiment.thresholds[threshold_index])
                self.pf1_dot.set_ydata(self._experiment.metrics[corruption_index][threshold_index].pairwise_f1)
                self.new_metrics_dot.set_xdata(self._experiment.thresholds[threshold_index])
                self.new_metrics_dot.set_ydata(self._experiment.new_metrics[corruption_index][threshold_index].
                                               net_expected_cost)
                self._figures[0].canvas.draw()
                self._figures[1].canvas.draw()
                self._threshold_index = threshold_index

    def __init__(self, number_entities, records_per_entity):
        ## Parameters ##
        self.corruption_multipliers = np.linspace(0, 0.1, 10)
        self.thresholds = np.linspace(0, 1, 10)
        ################
        uncorrupted_synthetic = Synthetic(number_entities, records_per_entity, number_features=2, sigma=0)
        self._uncorrupted_synthetic_train = uncorrupted_synthetic.sample_and_remove(float(number_entities) *
                                                                                    records_per_entity/2)
        self._train_pair_seed = generate_pair_seed(self._uncorrupted_synthetic_train.database,
                                                   self._uncorrupted_synthetic_train.labels, 300, balancing=True)
        self.uncorrupted_synthetic_test = uncorrupted_synthetic
        self._synthetic_train = list()
        self.synthetic_test = list()
        self.corruption_train = np.random.normal(loc=0.0, scale=1.0,
                                                 size=[len(self._uncorrupted_synthetic_train.database.records),
                                                       uncorrupted_synthetic.database.feature_descriptor.number])
        self.corruption_test = np.random.normal(loc=0.0, scale=1.0,
                                                size=[len(self.uncorrupted_synthetic_test.database.records),
                                                      uncorrupted_synthetic.database.feature_descriptor.number])
        for multiplier in self.corruption_multipliers:
            new_train = deepcopy(self._uncorrupted_synthetic_train)
            new_test = deepcopy(self.uncorrupted_synthetic_test)
            new_train.corrupt(multiplier*self.corruption_train)
            new_test.corrupt(multiplier*self.corruption_test)
            self._synthetic_train.append(new_train)
            self.synthetic_test.append(new_test)
        self.predicted_labels, self.metrics, self.er, self.new_metrics = self.run()

    def run(self):
        """
        Runs ER for all corruption levels and all thresholds
        :return predicted_labels: List of lists of predicted labels.
                                  predicted_labels[corruption_index][threshold_index] = dict [identifier, cluster label]
        :return metrics: List of lists of metric objects.
                         metrics[corruption_index][threshold_index] = Metrics object
        :return er_objects: List of EntityResolution objects.
                            er_objects[corruption_index][threshold_index] = EntityResolution
        :return new_metrics_objects: List of NewMetrics objects.
                                    new_metrics_objects[corruption_index][threshold_index] = NewMetrics
        """
        predicted_labels = list()
        metrics = list()
        er_objects = list()
        new_metrics_objects = list()
        for synthetic_train, synthetic_test in izip(self._synthetic_train, self.synthetic_test):
            er = EntityResolution()
            weak_match_function = er.train(synthetic_train.database, synthetic_train.labels, 300, balancing=True,
                                           pair_seed=self._train_pair_seed)
            metrics_sublist = list()
            labels_sublist = list()
            er_sublist = list()
            new_metrics_sublist = list()
            for threshold in self.thresholds:
                labels_pred = er.run(synthetic_test.database, weak_match_function, threshold, single_block=True,
                                     match_type='weak', max_block_size=np.Inf, cores=1)
                er_deepcopy = deepcopy(er)
                er_sublist.append(er_deepcopy)
                metrics_sublist.append(Metrics(synthetic_test.labels, labels_pred))
                new_metrics_sublist.append(NewMetrics(synthetic_test.database, er_deepcopy))
                labels_sublist.append(labels_pred)
            metrics.append(metrics_sublist)
            new_metrics_objects.append(new_metrics_sublist)
            predicted_labels.append(labels_sublist)
            er_objects.append(er_sublist)
        return predicted_labels, metrics, er_objects, new_metrics_objects

    def plot_metrics(self):
        """
        Makes precision/recall plots
        """
        pairwise_precision_array = np.empty((len(self.metrics), len(self.corruption_multipliers)))
        pairwise_recall_array = np.empty((len(self.metrics), len(self.corruption_multipliers)))
        pairwise_f1_array = np.empty((len(self.metrics), len(self.corruption_multipliers)))
        for threshold_index, metrics in enumerate(self.metrics):  # metrics at set threshold
            for corruption_index, metric in enumerate(metrics):  # metrics at set corruption
                pairwise_precision_array[threshold_index, corruption_index] = metric.pairwise_precision
                pairwise_recall_array[threshold_index, corruption_index] = metric.pairwise_recall
                pairwise_f1_array[threshold_index, corruption_index] = metric.pairwise_f1

        ## Precision vs. Recall
        plt.plot(pairwise_recall_array, pairwise_precision_array)
        plt.title('Pairwise Precision Recall')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(self.corruption_multipliers.astype(str), title='Corruption')
        plt.show()

        ## Precision v. Threshold
        plt.plot(self.thresholds, pairwise_precision_array)
        plt.title('Pairwise Precision')
        plt.xlabel('Threshold')
        plt.ylabel('Precision')
        plt.legend(self.corruption_multipliers.astype(str), title='Corruption')
        plt.show()

        ## Recall v. Threshold
        plt.plot(self.thresholds, pairwise_recall_array)
        plt.title('Pairwise Recall')
        plt.xlabel('Threshold')
        plt.ylabel('Recall')
        plt.legend(self.corruption_multipliers.astype(str), title='Corruption')
        plt.show()

        ## F1 v Threshold
        plt.plot(self.thresholds, pairwise_f1_array)
        plt.title('Pairwise F1')
        plt.xlabel('Threshold')
        plt.ylabel('F1')
        plt.legend(self.corruption_multipliers.astype(str), title='Corruption')
        plt.show()

        print 'Threshold (rows) vs. Corruption Level (Columns)'
        print 'Pairwise Precision'
        np.set_printoptions(precision=5, suppress=True)  # no scientific notation
        print pairwise_precision_array
        print 'Pairwise Recall'
        print pairwise_recall_array
        print 'Pairwise F1'
        print pairwise_f1_array


def main():
    experiment = SyntheticExperiment(10, 10)
    plot = experiment.ResultsPlot(experiment)
    #experiment.plot_metrics()

if __name__ == '__main__':
    cProfile.run('main()')
