__author__ = 'mbarnes1'
from database import Database, SyntheticDatabase
from copy import deepcopy
from entityresolution import EntityResolution
from pairwise_features import generate_pair_seed
from itertools import izip
import numpy as np
from metrics import Metrics, _cluster
import matplotlib.pyplot as plt
plt.ion()
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider
import cProfile
import matplotlib.cm as cm
from new_metrics import NewMetrics
import cPickle as pickle
import csv
import datetime
# Color-blind safe color palette
# Blue (Precision): #4477AA
# Magenta (Recall): #CC6677
# Tan (F1): #DDCC77


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

            # ER Precision/Recall Lower Bound Debugging
            fig = plt.figure()
            self._figures.append(fig)
            ax = fig.add_subplot(111)
            #self._axes.append(ax)
            TP_FP_match = list()  # TP_match + FP_match
            TP_FP_swoosh = list()  # TP_swoosh + FP_swoosh
            for threshold_index, threshold in enumerate(experiment.thresholds):
                TP_FP_match.append(experiment.new_metrics[self._corruption_index][threshold_index].TP_FP_match)
                TP_FP_swoosh.append(experiment.new_metrics[self._corruption_index][threshold_index].TP_FP_swoosh)
            self.TP_FP_match, = ax.plot(experiment.thresholds, TP_FP_match, label='TP + FP, match', linewidth=2, color='g')
            self.TP_FP_swoosh, = ax.plot(experiment.thresholds, TP_FP_swoosh, label='TP + FP, swoosh', linewidth=2, color='r', linestyle='--')
            plt.legend(loc='upper left')
            ax.set_xlabel('Threshold')
            ax.set_ylabel('Count')
            ax.set_title('Lower Bound Debugging')


            # Match function precision recall curve
            fig = plt.figure()
            self._figures.append(fig)
            ax = fig.add_subplot(111)
            self._axes.append(ax)
            match_thresholds = experiment.er[self._corruption_index][self._threshold_index]._match_function.roc.prob
            match_precision = experiment.er[self._corruption_index][self._threshold_index]._match_function.roc.precision  # threshold index should be arbitrary
            match_recall = experiment.er[self._corruption_index][self._threshold_index]._match_function.roc.recall  # threshold index should be arbitrary
            match_f1 = experiment.er[self._corruption_index][self._threshold_index]._match_function.roc.f1  # threshold index should be arbitrary
            self.match_precision, = ax.plot(match_thresholds, match_precision, label='Precision', linewidth=2, color='#4477AA')  # blue
            self.match_recall, = ax.plot(match_thresholds, match_recall, label='Recall', linewidth=2, color='#CC6677')  # Magenta
            self.match_f1, = ax.plot(match_thresholds, match_f1, label='F1', linewidth=2, color='#DDCC77')  # Tan
            plt.legend(loc='upper left')
            ax.set_xlabel('Threshold')
            ax.set_ylabel('Score')
            ax.set_title('Match Function Performance')
            ax.set_xlim([0, 1.0])
            ax.set_ylim([0, 1.0])

            # Pairwise F1
            fig = plt.figure()
            self._figures.append(fig)
            ax = fig.add_subplot(111)
            #ax.grid(linestyle='--')
            self._axes.append([ax])
            pairwise_f1 = list()
            pairwise_f1_lower_bound = list()
            pairwise_precision = list()
            pairwise_precision_lower_bound = list()
            pairwise_recall = list()
            pairwise_recall_lower_bound = list()
            for threshold_index, threshold in enumerate(experiment.thresholds):
                pairwise_f1.append(experiment.metrics[self._corruption_index][threshold_index].pairwise_f1)
                pairwise_precision.append(experiment.metrics[self._corruption_index][threshold_index].pairwise_precision)
                pairwise_recall.append(experiment.metrics[self._corruption_index][threshold_index].pairwise_recall)
                pairwise_precision_lower_bound.append(experiment.new_metrics[self._corruption_index][threshold_index].precision_lower_bound)
                pairwise_recall_lower_bound.append(experiment.new_metrics[self._corruption_index][threshold_index].recall_lower_bound)
                pairwise_f1_lower_bound.append(experiment.new_metrics[self._corruption_index][threshold_index].f1_lower_bound)
            self.pf1, = ax.plot(experiment.thresholds, pairwise_f1, label='F1', linewidth=2, color='#DDCC77', linestyle='--')
            self.pp, = ax.plot(experiment.thresholds, pairwise_precision, label='Precision', linewidth=2, color='#4477AA', linestyle='--')
            self.pr, = ax.plot(experiment.thresholds, pairwise_recall, label='Recall', linewidth=2, color='#CC6677', linestyle='--')
            #self.pf1_dot, = ax.plot(experiment.thresholds[self._threshold_index], pairwise_f1[self._threshold_index],
            #                        'bo', markersize=10, label='Operating Point')
            self.pf1_lower, = ax.plot(experiment.thresholds, pairwise_f1_lower_bound, label='F1 Lower Bound', linewidth=2, color='#DDCC77')
            self.pp_lower, = ax.plot(experiment.thresholds, pairwise_precision_lower_bound, label='Precision Lower Bound', linewidth=2, color='#4477AA')
            self.pr_lower, = ax.plot(experiment.thresholds, pairwise_recall_lower_bound, label='Recall Lower Bound', linewidth=2, color='#CC6677')
            plt.legend(loc='upper left')
            self._axes[1][0].set_xlabel('Threshold')
            self._axes[1][0].set_ylabel('Pairwise Metric')
            self._axes[1][0].set_title('Pairwise Metrics')
            self._axes[1][0].set_ylim([0, 1.0])

            # New metric
            # fig = plt.figure()
            # self._figures.append(fig)
            # ax = fig.add_subplot(111)
            # ax.grid(linestyle='--')
            # self._axes.append([ax])
            # new_metrics_expected = list()
            # new_metrics_best = list()
            # new_metrics_worst = list()
            # for threshold_index, threshold in enumerate(experiment.thresholds):
            #     new_metrics_expected.append(-1*experiment.new_metrics[self._corruption_index][threshold_index].net_expected_cost)
            #     new_metrics_best.append(-1*experiment.new_metrics[self._corruption_index][threshold_index].greedy_best_cost)
            #     new_metrics_worst.append(-1*experiment.new_metrics[self._corruption_index][threshold_index].greedy_worst_cost)
            # self.new_metrics_expected, = ax.plot(experiment.thresholds, new_metrics_expected, label='Expected')
            # self.new_metrics_best, = ax.plot(experiment.thresholds, new_metrics_best, label='Upper Greedy Bound')
            # self.new_metrics_worst, = ax.plot(experiment.thresholds, new_metrics_worst, label='Lower Greedy Bound')
            # plt.legend(handles=[self.new_metrics_expected, self.new_metrics_best, self.new_metrics_worst], loc='upper left')
            # # self.new_metrics_dot, = ax.plot(experiment.thresholds[self._threshold_index],
            # #                                 new_metrics_expected[self._threshold_index], 'bo', markersize=10,
            # #                                 label='Operating Point')
            # self._axes[2][0].set_xlabel('Threshold')
            # self._axes[2][0].set_ylabel('-New Metric')
            # self._axes[2][0].set_title('New Metric - Path Costs')

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
                # new_metrics_expected = list()
                # new_metrics_best = list()
                # new_metrics_worst = list()
                pairwise_f1 = list()
                pairwise_f1_lower_bound = list()
                pairwise_precision = list()
                pairwise_precision_lower_bound = list()
                pairwise_recall = list()
                pairwise_recall_lower_bound = list()
                match_precision = self._experiment.er[self._corruption_index][self._threshold_index]._match_function.roc.precision
                match_recall = self._experiment.er[self._corruption_index][self._threshold_index]._match_function.roc.recall
                match_f1 = self._experiment.er[self._corruption_index][self._threshold_index]._match_function.roc.f1
                for threshold_index, threshold in enumerate(self._experiment.thresholds):
                    pairwise_f1.append(self._experiment.metrics[self._corruption_index][threshold_index].pairwise_f1)
                    pairwise_precision.append(self._experiment.metrics[self._corruption_index][threshold_index].pairwise_precision)
                    pairwise_recall.append(self._experiment.metrics[self._corruption_index][threshold_index].pairwise_recall)
                    pairwise_precision_lower_bound.append(self._experiment.new_metrics[self._corruption_index][threshold_index].precision_lower_bound)
                    pairwise_recall_lower_bound.append(self._experiment.new_metrics[self._corruption_index][threshold_index].recall_lower_bound)
                    pairwise_f1_lower_bound.append(self._experiment.new_metrics[self._corruption_index][threshold_index].f1_lower_bound)
                    # new_metrics_expected.append(-1*self._experiment.new_metrics[self._corruption_index][threshold_index].net_expected_cost)
                    # new_metrics_best.append(-1*self._experiment.new_metrics[self._corruption_index][threshold_index].greedy_best_cost)
                    # new_metrics_worst.append(-1*self._experiment.new_metrics[self._corruption_index][threshold_index].greedy_worst_cost)
                self.match_precision.set_ydata(match_precision)
                self.match_recall.set_ydata(match_recall)
                self.match_f1.set_ydata(match_f1)
                self.pf1.set_ydata(pairwise_f1)
                self.pf1_lower.set_ydata(pairwise_f1_lower_bound)
                self.pp.set_ydata(pairwise_precision)
                self.pp_lower.set_ydata(pairwise_precision_lower_bound)
                self.pr.set_ydata(pairwise_recall)
                self.pr_lower.set_ydata(pairwise_recall_lower_bound)
                #self.pf1_dot.set_ydata(pairwise_f1[self._threshold_index])
                # self.new_metrics_expected.set_ydata(new_metrics_expected)
                # self.new_metrics_best.set_ydata(new_metrics_best)
                # self.new_metrics_worst.set_ydata(new_metrics_worst)
                #self.new_metrics_dot.set_ydata(new_metrics[self._threshold_index])
                for figure in self._figures:
                    figure.canvas.draw()
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
                #self.pf1_dot.set_xdata(self._experiment.thresholds[threshold_index])
                #self.pf1_dot.set_ydata(self._experiment.metrics[corruption_index][threshold_index].pairwise_f1)
                #self.new_metrics_dot.set_xdata(self._experiment.thresholds[threshold_index])
                #self.new_metrics_dot.set_ydata(self._experiment.new_metrics[corruption_index][threshold_index].
                #                               net_expected_cost)
                for figure in self._figures:
                    figure.canvas.draw()
                self._threshold_index = threshold_index

    def __init__(self, number_entities, records_per_entity, train_database_size, validation_database_size,
                 train_class_balance, number_thresholds):
        ## Parameters ##
        self.corruption_multipliers = np.array([0.025])  # np.linspace(0, 0.025, 5)
        self.thresholds = np.linspace(0, 1, number_thresholds)
        ################
        uncorrupted_synthetic = SyntheticDatabase(number_entities, records_per_entity, number_features=2)
        self._uncorrupted_synthetic_train = uncorrupted_synthetic.sample_and_remove(train_database_size)
        self._uncorrupted_synthetic_validation = uncorrupted_synthetic.sample_and_remove(validation_database_size)
        self.uncorrupted_synthetic_test = uncorrupted_synthetic
        self._train_pair_seed = generate_pair_seed(self._uncorrupted_synthetic_train.database,
                                                   self._uncorrupted_synthetic_train.labels, train_class_balance)
        self._synthetic_train = list()
        self.synthetic_validation = list()
        self.synthetic_test = list()
        self.corruption_train = np.random.normal(loc=0.0, scale=1.0,
                                                 size=[train_database_size,
                                                 uncorrupted_synthetic.database.feature_descriptor.number])
        self.corruption_validation = np.random.normal(loc=0.0, scale=1.0,
                                                      size=[validation_database_size,
                                                      uncorrupted_synthetic.database.feature_descriptor.number])
        self.corruption_test = np.random.normal(loc=0.0, scale=1.0,
                                                size=[len(self.uncorrupted_synthetic_test.database.records),
                                                uncorrupted_synthetic.database.feature_descriptor.number])
        for multiplier in self.corruption_multipliers:
            new_train = deepcopy(self._uncorrupted_synthetic_train)
            new_validation = deepcopy(self._uncorrupted_synthetic_validation)
            new_test = deepcopy(self.uncorrupted_synthetic_test)
            new_train.corrupt(multiplier*self.corruption_train)
            new_validation.corrupt(multiplier*self.corruption_validation)
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
            weak_match_function = er.train(synthetic_train.database, synthetic_train.labels, self._train_pair_seed)
            print 'Testing pairwise match function precision and recall on validation database'
            roc = weak_match_function.test(synthetic_test.database, synthetic_test.labels, 0.5)
            class_balance_test = get_pairwise_class_balance(synthetic_test.labels)
            #roc.make_plot()
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
                new_metrics_sublist.append(NewMetrics(synthetic_test.database, er_deepcopy, class_balance_test))
                labels_sublist.append(labels_pred)
            metrics.append(metrics_sublist)
            new_metrics_objects.append(new_metrics_sublist)
            predicted_labels.append(labels_sublist)
            er_objects.append(er_sublist)
        return predicted_labels, metrics, er_objects, new_metrics_objects

    # def plot_metrics(self):
    #     """
    #     Makes precision/recall plots
    #     """
    #     pairwise_precision_array = np.empty((len(self.metrics), len(self.corruption_multipliers)))
    #     pairwise_recall_array = np.empty((len(self.metrics), len(self.corruption_multipliers)))
    #     pairwise_f1_array = np.empty((len(self.metrics), len(self.corruption_multipliers)))
    #     for threshold_index, metrics in enumerate(self.metrics):  # metrics at set threshold
    #         for corruption_index, metric in enumerate(metrics):  # metrics at set corruption
    #             pairwise_precision_array[threshold_index, corruption_index] = metric.pairwise_precision
    #             pairwise_recall_array[threshold_index, corruption_index] = metric.pairwise_recall
    #             pairwise_f1_array[threshold_index, corruption_index] = metric.pairwise_f1
    #
    #     ## Precision vs. Recall
    #     plt.plot(pairwise_recall_array, pairwise_precision_array)
    #     plt.title('Pairwise Precision Recall')
    #     plt.xlabel('Recall')
    #     plt.ylabel('Precision')
    #     plt.legend(self.corruption_multipliers.astype(str), title='Corruption')
    #     plt.show()
    #
    #     ## Precision v. Threshold
    #     plt.plot(self.thresholds, pairwise_precision_array)
    #     plt.title('Pairwise Precision')
    #     plt.xlabel('Threshold')
    #     plt.ylabel('Precision')
    #     plt.legend(self.corruption_multipliers.astype(str), title='Corruption')
    #     plt.show()
    #
    #     ## Recall v. Threshold
    #     plt.plot(self.thresholds, pairwise_recall_array)
    #     plt.title('Pairwise Recall')
    #     plt.xlabel('Threshold')
    #     plt.ylabel('Recall')
    #     plt.legend(self.corruption_multipliers.astype(str), title='Corruption')
    #     plt.show()
    #
    #     ## F1 v Threshold
    #     plt.plot(self.thresholds, pairwise_f1_array)
    #     plt.title('Pairwise F1')
    #     plt.xlabel('Threshold')
    #     plt.ylabel('F1')
    #     plt.legend(self.corruption_multipliers.astype(str), title='Corruption')
    #     plt.show()
    #
    #     print 'Threshold (rows) vs. Corruption Level (Columns)'
    #     print 'Pairwise Precision'
    #     np.set_printoptions(precision=5, suppress=True)  # no scientific notation
    #     print pairwise_precision_array
    #     print 'Pairwise Recall'
    #     print pairwise_recall_array
    #     print 'Pairwise F1'
    #     print pairwise_f1_array


class Experiment(object):
    """
    An experiment on a single synthetic or real database, with varying cutoff thresholds
    """
    def __init__(self, database_train, database_validation, database_test, labels_train, labels_validation, labels_test,
                 train_class_balance, thresholds):
        """
        Performs entity resolution on a database at varying thresholds
        :param database_train: Database object for training match function
        :param database_validation: Database object for estimating match precision/recall performance
        :param database_test: Database object for testing entity resolution
        :param labels_train: A dictionary of the true labels [record id, label]
        :param labels_validation: A dictionary of the true labels [record id, label]
        :param labels_test: A dictionary of the true labels [record id, label]
        :param train_class_balance: Float [0, 1.0]. Train with this percent of positive samples
        :param thresholds: List of thresholds to run ER at
        """
        self._database_train = database_train
        self._database_validation = database_validation
        self._database_test = database_test
        self._labels_train = labels_train
        self._labels_validation = labels_validation
        self._labels_test = labels_test
        self._train_class_balance = train_class_balance
        self.thresholds = thresholds
        print 'Generating pairwise seed for training database'
        self._train_pair_seed = generate_pair_seed(self._database_train, self._labels_train, train_class_balance)
        self._predicted_labels, self.metrics, self.new_metrics = self.run()

    def run(self):
        """
        Runs ER at all thresholds
        :return predicted_labels: List of lists of predicted labels.
                                  predicted_labels[threshold_index] = dict [identifier, cluster label]
        :return metrics: List of lists of metric objects.
                         metrics[threshold_index] = Metrics object
        :return er_objects: List of EntityResolution objects.
                            er_objects[threshold_index] = EntityResolution
        :return new_metrics_objects: List of NewMetrics objects.
                                    new_metrics_objects[threshold_index] = NewMetrics
        """
        er = EntityResolution()
        weak_match_function = er.train(self._database_train, self._labels_train, self._train_pair_seed)
        print 'Testing pairwise match function on test database'
        ROC = weak_match_function.test(self._database_validation, self._labels_validation, 0.5)
        #ROC.make_plot()
        metrics_list = list()
        labels_list = list()
        new_metrics_list = list()
        class_balance_test = get_pairwise_class_balance(self._labels_test)
        for threshold in self.thresholds:
            print 'Running entity resolution at threshold =', threshold
            labels_pred = er.run(self._database_test, weak_match_function, threshold, single_block=True,
                                 match_type='weak', max_block_size=np.Inf, cores=1)
            metrics_list.append(Metrics(self._labels_test, labels_pred))
            new_metrics_list.append(NewMetrics(self._database_test, deepcopy(er), class_balance_test))
            labels_list.append(labels_pred)
        return labels_list, metrics_list, new_metrics_list

    def plot(self):
        """
        Plot of the metrics at varying thresholds
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

        pairwise_f1 = list()
        pairwise_f1_lower_bound = list()
        pairwise_precision = list()
        pairwise_precision_lower_bound = list()
        pairwise_recall = list()
        pairwise_recall_lower_bound = list()
        pairwise_precision_lower_bound_lower_ci = list()
        pairwise_precision_lower_bound_upper_ci = list()
        pairwise_recall_lower_bound_lower_ci = list()
        pairwise_recall_lower_bound_upper_ci = list()
        matches = list()  # TP_match + FP_match
        pairs_r = list()  # TP_swoosh + FP_swoosh
        for threshold_index, threshold in enumerate(self.thresholds):
            pairwise_f1.append(self.metrics[threshold_index].pairwise_f1)
            pairwise_precision.append(self.metrics[threshold_index].pairwise_precision)
            pairwise_recall.append(self.metrics[threshold_index].pairwise_recall)
            pairwise_precision_lower_bound.append(self.new_metrics[threshold_index].precision_lower_bound)
            pairwise_precision_lower_bound_lower_ci.append(self.new_metrics[threshold_index].precision_lower_bound_lower_ci)
            pairwise_precision_lower_bound_upper_ci.append(self.new_metrics[threshold_index].precision_lower_bound_upper_ci)
            pairwise_recall_lower_bound.append(self.new_metrics[threshold_index].recall_lower_bound)
            pairwise_recall_lower_bound_lower_ci.append(self.new_metrics[threshold_index].recall_lower_bound_lower_ci)
            pairwise_recall_lower_bound_upper_ci.append(self.new_metrics[threshold_index].recall_lower_bound_upper_ci)
            pairwise_f1_lower_bound.append(self.new_metrics[threshold_index].f1_lower_bound)
            matches.append(self.new_metrics[threshold_index].TP_FP_match)
            pairs_r.append(self.new_metrics[threshold_index].TP_FP_swoosh)
        ## Save results to text files
        filename = '../figures/'+timestamp+'_data.csv'
        with open(filename, 'wb') as f:
            f.write('Threshold, Pairwise precision, Pairwise Precision Lower Bound, Lower CI, Upper CI,'
                    ' Pairwise Recall, Pairwise Recall Lower Bound, Lower CI, Upper CI, '
                    'Pairwise F1, Pairwise F1 lower bound, |M|, |Pairs(R)|\n')
            writer = csv.writer(f)
            rows = zip(self.thresholds, pairwise_precision, pairwise_precision_lower_bound,
                       pairwise_precision_lower_bound_lower_ci, pairwise_precision_lower_bound_upper_ci,
                       pairwise_recall, pairwise_recall_lower_bound, pairwise_recall_lower_bound_lower_ci,
                       pairwise_recall_lower_bound_upper_ci, pairwise_f1, pairwise_f1_lower_bound, matches, pairs_r)
            for row in rows:
                writer.writerow(row)
        f.close()

        ## Plot the results
        save_as = '../figures/'+timestamp
        plot_results(filename, save_as)


def plot_size_experiment(filename, save_as=None):
    """
    Plots degradation in performance as database size increases
    :param filename: Path to file
    :param save_as: Save file name (optional)
    """
    figsize = (5, 5.0*3/4)
    label_font_size = 10
    legend_font_size = 10
    title_font_size = 12

    print 'Loading csv results file...'
    data = np.loadtxt(open(filename, 'rb'), delimiter=',', skiprows=1)
    sizes = data[:, 0]
    precision_small = data[:, 1]
    recall_small = data[:, 2]
    f1_small = data[:, 3]
    precision_large = data[:, 4]
    precision_large_bound = data[:, 5]
    precision_large_bound_lower_ci = data[:, 6]
    precision_large_bound_upper_ci = data[:, 7]
    recall_large = data[:, 8]
    recall_large_bound = data[:, 9]
    recall_large_bound_lower_ci = data[:, 10]
    recall_large_bound_upper_ci = data[:, 11]

    ## Precision
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ci_legend_proxy = Rectangle((0.000001, 0.000001), 1.0, 1., fc="#BFBFBF")
    ax.fill_between(sizes, precision_large_bound_lower_ci, precision_large_bound_upper_ci, color='#BFBFBF', linewidth=0.0)
    p1, = ax.plot(sizes, precision_small, label='Original', linewidth=2, color='k', linestyle='--')
    p1.set_dashes([10, 6])
    p2, = ax.plot(sizes, precision_large, label='Optimized, true', linewidth=2, color='k')
    p2.set_dashes([2, 4])
    p3, = ax.plot(sizes, precision_large_bound, label='Optimized Lower Bound', linewidth=2, color='k')
    ax.set_ylim([0, 1.05])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xticks([100, 300, 500, 700, 900])
    lg = plt.legend([p1, p3, ci_legend_proxy, p2], ['Original', 'Optimized Lower Bound', 'Bound 95% CI', 'Optimized, true'], loc='lower right', fontsize=legend_font_size, handlelength=3)
    ax.set_xlabel('Database Size', fontsize=label_font_size)
    ax.set_ylabel('Precision', fontsize=label_font_size)
    ax.set_title('Precision Degradation', fontsize=title_font_size)
    plt.tight_layout()
    if save_as is not None:
        output = save_as + '_precision.eps'
        fig.savefig(output, bbox_inches='tight')

    ## Recall
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ci_legend_proxy = Rectangle((0.000001, 0.000001), 1.0, 1., fc="#BFBFBF")
    ax.fill_between(sizes, recall_large_bound_lower_ci, recall_large_bound_upper_ci, color='#BFBFBF', linewidth=0.0)
    p1, = ax.plot(sizes, recall_small, label='Original', linewidth=2, color='k', linestyle='--')
    p2, = ax.plot(sizes, recall_large, label='Optimized, true', linewidth=2, color='k', linestyle=':')
    p3, = ax.plot(sizes, recall_large_bound, label='Optimized Lower Bound', linewidth=2, color='k')
    ax.set_ylim([0, 1.05])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xticks([100, 300, 500, 700, 900])
    plt.legend([p1, p3, ci_legend_proxy, p2], ['Original', 'Optimized Lower Bound', 'Bound 95% CI', 'Optimized, true'], loc='lower right', fontsize=legend_font_size)
    ax.set_xlabel('Database Size', fontsize=label_font_size)
    ax.set_ylabel('Recall', fontsize=label_font_size)
    ax.set_title('Recall Degradation', fontsize=title_font_size)
    plt.tight_layout()
    if save_as is not None:
        output = save_as + '_recall.eps'
        fig.savefig(output, bbox_inches='tight')

    plt.show()


def plot_results(filename, save_as=None):
    """
    Plots results saved in filename
    :param filename: Path to file
    :param save_as: String. Save figures if not None
    """
    figsize = (5, 5.0*3/4)
    label_font_size = 10
    legend_font_size = 10
    title_font_size = 12

    print 'Loading csv results file...'
    data = np.loadtxt(open(filename, 'rb'), delimiter=',', skiprows=1)
    thresholds = data[:, 0]
    precision = data[:, 1]
    precision_bound = data[:, 2]
    precision_bound_lower_ci = data[:, 3]
    precision_bound_upper_ci = data[:, 4]
    recall = data[:, 5]
    recall_bound = data[:, 6]
    recall_bound_lower_ci = data[:, 7]
    recall_bound_upper_ci = data[:, 8]
    f1 = data[:, 9]
    f1_bound = data[:, 10]
    n_matches = data[:, 11]
    pairs_r = data[:, 12]

    print 'Plotting experimental results...'
    # Lemma 1
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(thresholds, n_matches, label='M', linewidth=2, color='k')
    ax.plot(thresholds, pairs_r, label='Pairs(R)', linewidth=2, color='k', linestyle=':')
    plt.legend(loc='upper right', fontsize=legend_font_size)
    ax.set_xlabel('Threshold', fontsize=label_font_size)
    ax.set_ylabel('Set Size', fontsize=label_font_size)
    ax.set_title('Lemma 1 Analysis', fontsize=title_font_size)
    plt.tight_layout()
    if save_as is not None:
        filename = save_as+'_lemma1.eps'
        fig.savefig(filename, bbox_inches='tight')

    # Pairwise Precision
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    #ax.grid(linestyle='--')
    ci_legend_proxy = Rectangle((0, 0), 1, 1, fc="#BFBFBF")
    ax.fill_between(thresholds, precision_bound_lower_ci, precision_bound_upper_ci, color='#BFBFBF', linewidth=0.0)
    ax1, = ax.plot(thresholds, precision, label='True Precision', linewidth=2, color='k', linestyle=':')  #4477AA
    ax1.set_dashes([2, 4])
    ax2, = ax.plot(thresholds, precision_bound, label='Estimated Lower Bound', linewidth=2, color='k')  #4477AA
    plt.legend([ax1, ax2, ci_legend_proxy], ['True Precision', 'Estimated Lower Bound', '95% CI'], loc='upper left', fontsize=legend_font_size, handlelength=3)
    ax.set_xlabel('Match Threshold', fontsize=label_font_size)
    ax.set_ylabel('Pairwise Precision', fontsize=label_font_size)
    #ax.set_title('Pairwise Precision', fontsize=title_font_size)
    ax.set_ylim([0, 1.05])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.tight_layout()
    if save_as is not None:
        filename = save_as + '_precision.eps'
        fig.savefig(filename, bbox_inches='tight')

    # Pairwise Recall
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ci_legend_proxy = Rectangle((0, 0), 1, 1, fc="#BFBFBF")
    ax.fill_between(thresholds, recall_bound_lower_ci, recall_bound_upper_ci, color='#BFBFBF', linewidth=0.0, label='95% CI')
    ax1, = ax.plot(thresholds, recall, label='True Recall', linewidth=2, color='k', linestyle=':')  #'#CC6677'
    ax1.set_dashes([2, 4])
    ax2, = ax.plot(thresholds, recall_bound, label='Estimated Lower Bound', linewidth=2, color='k')  #'#CC6677'
    plt.legend([ax1, ax2, ci_legend_proxy], ['True Recall', 'Estimated Lower Bound', '95% CI'], loc='lower left', fontsize=legend_font_size, handlelength=3)
    ax.set_xlabel('Match Threshold', fontsize=label_font_size)
    ax.set_ylabel('Pairwise Recall', fontsize=label_font_size)
    #ax.set_title('Pairwise Recall', fontsize=title_font_size)
    ax.set_ylim([0, 1.05])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.tight_layout()
    if save_as is not None:
        filename = save_as+'_recall.eps'
        fig.savefig(filename, bbox_inches='tight')
    print 'Finished plotting'
    plt.show()


def get_pairwise_class_balance(labels):
    """
    Returns the percent of positive pairs out of all the pairs in database.
    Eventually this should be automated with density estimates, using a train database, train labels, and test database
    :param labels: Corresponding labels for the database object. Dict of [record id, label]
    :return class_balance: Percent of positive pairs in database, [0, 1.0]
    """
    print 'Calculating class balance for labels:'
    print(labels)
    number_records = len(labels)

    total_number_pairs = number_records*(number_records-1)/2
    print '     Total number of pairs:', total_number_pairs
    clusters = _cluster(labels)
    total_number_positive_pairs = 0.0
    for cluster in clusters:
        number_cluster_pairs = len(cluster)*(len(cluster)-1)/2
        total_number_positive_pairs += number_cluster_pairs
    print '     Number of positive pairs:', total_number_positive_pairs
    class_balance = total_number_positive_pairs/total_number_pairs
    print '     Class balance:', class_balance
    return class_balance


def main():
    #### Real Experiment ####
    number_thresholds = 30
    dataset_name = 'trafficking'  # synthetic, synthetic_2d, synthetic_sizes, restaurant, abt-buy, trafficking

    if dataset_name == 'synthetic_2d':
        number_entities = 10
        records_per_entity = 30
        train_database_size = 100
        train_class_balance = 0.5
        validation_database_size = 100
        synthetic_experiment = SyntheticExperiment(number_entities, records_per_entity,
                                                   train_database_size, validation_database_size,
                                                   train_class_balance, number_thresholds)
        synthetic_plot = synthetic_experiment.ResultsPlot(synthetic_experiment)
        #pickle.dump(synthetic_experiment, open('synthetic_experiment.p', 'wb'))
    elif dataset_name == 'synthetic_sizes':  # synthetic datasets at various sizes, to show degredation of performance
        resolution = 88
        number_features = 10
        number_entities = np.linspace(10, 100, num=resolution)
        number_entities = number_entities.astype(int)
        records_per_entity = 10
        #train_database_size = 100
        train_class_balance = 0.5
        #validation_database_size = 100
        corruption_multiplier = .001


        databases = list()
        db = SyntheticDatabase(number_entities[0], records_per_entity, number_features=number_features)
        databases.append(deepcopy(db))
        add_entities = [x - number_entities[i - 1] for i, x in enumerate(number_entities)][1:]
        for add in add_entities:
            db.add(add, records_per_entity)
            databases.append(deepcopy(db))
        corruption = np.random.normal(loc=0.0, scale=1.0, size=[number_entities[-1]*records_per_entity, number_features])
        train = deepcopy(databases[0])
        validation = deepcopy(databases[0])
        train.corrupt(corruption_multiplier*np.random.normal(loc=0.0, scale=1.0, size=[len(train.database.records), number_features]))
        validation.corrupt(corruption_multiplier*np.random.normal(loc=0.0, scale=1.0, size=[len(train.database.records), number_features]))
        for db in databases:
            db.corrupt(corruption_multiplier*corruption[:len(db.database.records), :])
        er = EntityResolution()
        train_pair_seed = generate_pair_seed(train.database, train.labels, train_class_balance)
        weak_match_function = er.train(train.database, train.labels, train_pair_seed)
        ROC = weak_match_function.test(validation.database, validation.labels, 0.5)
        #ROC.make_plot()

        ## Optimize ER on small dataset
        thresholds = np.linspace(0, 1.0, 10)
        metrics_list = list()
        #new_metrics_list = list()
        pairwise_precision = list()
        pairwise_recall = list()
        pairwise_f1 = list()
        for threshold in thresholds:
            labels_pred = er.run(deepcopy(databases[0].database), weak_match_function, threshold, single_block=True,
                                 match_type='weak', max_block_size=np.Inf, cores=1)
            met = Metrics(databases[0].labels, labels_pred)
            metrics_list.append(met)
            pairwise_precision.append(met.pairwise_precision)
            pairwise_recall.append(met.pairwise_recall)
            pairwise_f1.append(met.pairwise_f1)
            #class_balance_test = get_pairwise_class_balance(databases[0].labels)
            #new_metrics_list.append(NewMetrics(databases[0].database, er, class_balance_test))
        plt.plot(thresholds, pairwise_precision, label='Precision')
        plt.plot(thresholds, pairwise_recall, label='Recall')
        plt.plot(thresholds, pairwise_f1, label='F1')
        plt.xlabel('Threshold')
        plt.legend()
        plt.ylabel('Score')
        plt.title('Optimizing ER on small dataset')
        #i = np.argmax(np.array(pairwise_f1))
        #small_optimal_threshold = thresholds[i]  # optimize this
        small_optimal_threshold = 0.6
        print 'Optimal small threshold set at =', small_optimal_threshold
        plt.show()

        ## Possible score by optimizing on larger dataset
        metrics_list = list()
        pairwise_precision = list()
        pairwise_recall = list()
        pairwise_f1 = list()
        thresholds_largedataset = np.linspace(0.6, 1.0, 8)
        precision_lower_bound = list()
        recall_lower_bound = list()
        f1_lower_bound = list()
        for threshold in thresholds_largedataset:
            labels_pred = er.run(deepcopy(databases[-1].database), weak_match_function, threshold, single_block=True,
                                 match_type='weak', max_block_size=np.Inf, cores=1)
            met = Metrics(databases[-1].labels, labels_pred)
            metrics_list.append(met)
            pairwise_precision.append(met.pairwise_precision)
            pairwise_recall.append(met.pairwise_recall)
            pairwise_f1.append(met.pairwise_f1)
            class_balance_test = get_pairwise_class_balance(databases[-1].labels)
            new_metric = NewMetrics(databases[-1].database, er, class_balance_test)
            precision_lower_bound.append(new_metric.precision_lower_bound)
            recall_lower_bound.append(new_metric.recall_lower_bound)
            f1_lower_bound.append(new_metric.f1_lower_bound)
        plt.plot(thresholds_largedataset, pairwise_precision, label='Precision', color='r')
        plt.plot(thresholds_largedataset, pairwise_recall, label='Recall', color='b')
        plt.plot(thresholds_largedataset, pairwise_f1, label='F1', color='g')
        plt.plot(thresholds_largedataset, precision_lower_bound, label='Precision Bound', color='r', linestyle=':')
        plt.plot(thresholds_largedataset, recall_lower_bound, label='Recall Bound', color='b', linestyle=':')
        plt.plot(thresholds_largedataset, f1_lower_bound, label='F1 Bound', color='g', linestyle=':')
        i = np.argmax(np.array(f1_lower_bound))
        large_optimal_threshold = thresholds_largedataset[i]
        print 'Optimal large threshold automatically set at =', large_optimal_threshold
        print 'If not correct: debug.'
        plt.xlabel('Threshold')
        plt.legend()
        plt.ylabel('Score')
        plt.title('Optimizing ER on large dataset')
        plt.show()

        ## Run on all dataset sizes
        #new_metrics_list = list()
        database_sizes = list()
        small_pairwise_precision = list()
        small_pairwise_recall = list()
        small_pairwise_f1 = list()
        large_precision_bound = list()
        large_precision_bound_lower_ci = list()
        large_precision_bound_upper_ci = list()
        large_precision = list()
        large_recall_bound = list()
        large_recall_bound_lower_ci = list()
        large_recall_bound_upper_ci = list()
        large_recall = list()
        large_f1 = list()
        large_f1_bound = list()
        for db in databases:
            print 'Analyzing synthetic database with', len(db.database.records), 'records'
            database_sizes.append(len(db.database.records))
            labels_pred = er.run(db.database, weak_match_function, small_optimal_threshold, single_block=True, match_type='weak',
                                 max_block_size=np.Inf, cores=1)
            met = Metrics(db.labels, labels_pred)
            small_pairwise_precision.append(met.pairwise_precision)
            small_pairwise_recall.append(met.pairwise_recall)
            small_pairwise_f1.append(met.pairwise_f1)

            labels_pred = er.run(db.database, weak_match_function, large_optimal_threshold, single_block=True, match_type='weak',
                                 max_block_size=np.Inf, cores=1)
            met = Metrics(db.labels, labels_pred)
            large_precision.append(met.pairwise_precision)
            large_recall.append(met.pairwise_recall)
            large_f1.append(met.pairwise_f1)
            class_balance_test = get_pairwise_class_balance(db.labels)
            new_metric = NewMetrics(db.database, er, class_balance_test)
            large_precision_bound.append(new_metric.precision_lower_bound)
            large_recall_bound.append(new_metric.recall_lower_bound)
            large_f1_bound.append(new_metric.f1_lower_bound)
            large_precision_bound_lower_ci.append(new_metric.precision_lower_bound_lower_ci)
            large_precision_bound_upper_ci.append(new_metric.precision_lower_bound_upper_ci)
            large_recall_bound_lower_ci.append(new_metric.recall_lower_bound_lower_ci)
            large_recall_bound_upper_ci.append(new_metric.recall_lower_bound_upper_ci)

        with open('synthetic_sizes.csv', 'wb') as f:
            f.write('Database size, Precision (small opt), Recall (small opt), F1 (small opt), Precision (large opt), Precision bound (large opt), Lower CI, Upper CI, Recall (large opt), Recall bound (large opt), Lower CI, Upper CI, F1 (large opt), F1 bound (large opt)\n')
            writer = csv.writer(f)
            writer.writerows(izip(database_sizes, small_pairwise_precision, small_pairwise_recall, small_pairwise_f1, large_precision, large_precision_bound, large_precision_bound_lower_ci, large_precision_bound_upper_ci, large_recall, large_recall_bound, large_recall_bound_lower_ci, large_recall_bound_upper_ci, large_f1, large_f1_bound))
        f.close()
        plt.figure()
        plt.plot(database_sizes, pairwise_precision, label='Precision', color='#4477AA', linewidth=3)
        plt.plot(database_sizes, pairwise_recall, label='Recall', color='#CC6677', linewidth=3)
        #plt.plot(database_sizes, pairwise_f1, label='F1', color='#DDCC77', linewidth=2)
        plt.ylim([0, 1.05])
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.legend(title='Pairwise:', loc='lower left')
        plt.xlabel('Number of Records')
        plt.ylabel('Pairwise Score')
        plt.title('Performance Degredation')
        plt.show()
    elif dataset_name == 'synthetic':
        number_entities = 100
        records_per_entity = 10
        train_database_size = 200
        train_class_balance = 0.5
        validation_database_size = 200
        corruption = 0.001  #0.025
        number_thresholds = 30
        number_features = 10

        synthetic_database = SyntheticDatabase(number_entities, records_per_entity, number_features=number_features)
        corruption_array = corruption*np.random.normal(loc=0.0, scale=1.0, size=[validation_database_size,
                                                       synthetic_database.database.feature_descriptor.number])
        synthetic_database.corrupt(corruption_array)
        synthetic_train = synthetic_database.sample_and_remove(train_database_size)
        synthetic_validation = synthetic_database.sample_and_remove(validation_database_size)
        synthetic_test = synthetic_database
        thresholds = np.linspace(0, 1, number_thresholds)
        experiment = Experiment(synthetic_train.database, synthetic_validation.database, synthetic_test.database,
                                synthetic_train.labels, synthetic_validation.labels, synthetic_test.labels,
                                train_class_balance, thresholds)
        experiment.plot()
    else:
        if dataset_name == 'restaurant':  # 864 records, 112 matches
            features_path = '../data/restaurant/merged.csv'
            labels_path = '../data/restaurant/labels.csv'
            train_database_size = 300
            train_class_balance = .4
            validation_database_size = 200
            database = Database(annotation_path=features_path)
        elif dataset_name == 'abt-buy':  # ~4900 records, 1300 matches
            features_path = '../data/Abt-Buy/merged.csv'
            labels_path = '../data/Abt-Buy/labels.csv'
            train_database_size = 300
            train_class_balance = 0.4
            validation_database_size = 300
            database = Database(annotation_path=features_path)
        elif dataset_name == 'trafficking':
            features_path = '../data/trafficking/features.csv'
            labels_path = '../data/trafficking/labels.csv'
            train_database_size = 3000
            train_class_balance = 0.4
            validation_database_size = 3000
            database = Database(annotation_path=features_path)
        else:
            raise Exception('Invalid dataset name')
        thresholds = np.linspace(0, 1, number_thresholds)
        labels = np.loadtxt(open(labels_path, 'rb'))
        database_train = database.sample_and_remove(train_database_size)
        database_validation = database.sample_and_remove(validation_database_size)
        database_test = database
        labels_train = dict()
        labels_validation = dict()
        labels_test = dict()
        for identifier, label in enumerate(labels):
            if identifier in database_train.records:
                labels_train[identifier] = label
            elif identifier in database_validation.records:
                labels_validation[identifier] = label
            elif identifier in database_test.records:
                labels_test[identifier] = label
            else:
                raise Exception('Record identifier ' + str(identifier) + ' not in either database')

        experiment = Experiment(database_train, database_validation, database_test,
                                labels_train, labels_validation, labels_test,
                                train_class_balance, thresholds)
        #print 'Saving results'
        #pickle.dump(experiment, open('experiment.p', 'wb'))
        experiment.plot()
    print 'Finished'

if __name__ == '__main__':
    cProfile.run('main()')
