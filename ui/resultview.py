from itertools import count

import pandas as pd
from PyQt6.QtWidgets import QWidget
from PyQt6 import uic
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import random

from sklearn.preprocessing import MinMaxScaler

from utilities.classifier import Classifier
from utilities.feature_strength import FeatureStrength
from utilities.mlp_classifier import MLPClassifier
from utilities.nb_classifier import NBClassifier
from utilities.network import Network
from utilities.pandas_model import PandasModel
from utilities.rf_classifier import RFClassifier
from utilities.svmp_classifier import SVMPClassifier
from utilities.svmr_classifier import SVMRClassifier


class ResultView(QWidget):

    def __init__(self, data, name, parent=None):
        super(ResultView, self).__init__(parent)
        uic.loadUi('ui/result_view.ui', self)
        self.dataset_name = name
        self.data = data
        self.classified_data = None
        self.network = nx.DiGraph()

        self.combo_class.clear()
        self.combo_class.addItems(self.data.columns)
        self.current_class = self.data.columns[0]
        self.combo_class.currentTextChanged.connect(self._on_class_combobox_changed)

        self.models = ["NB", "SVMP", "SVMR", "MLP", "RF"]
        self.combo_models.clear()
        self.combo_models.addItems(self.models)
        self.combo_models.currentTextChanged.connect(self._on_model_combobox_changed)
        self.current_model = self.models[0]

        self.button_run.clicked.connect(self._run_classifier)
       #self.button_run.clicked.connect(self._test)

        self.setWindowTitle(name)

    def _on_model_combobox_changed(self, value):
        self.current_model = value

    def _on_class_combobox_changed(self, value):
        self.current_class = value

    def _run_classifier(self):
        classifier = Classifier(self.data, self.current_model, self.current_class)
        accuracy, self.classified_data, self.feature_names = classifier.run()

        self.label_accuracy.setText(f"Accuracy: {accuracy}")

        self._build_network()

    def _build_network(self):
        self.network = Network(self.classified_data).build_network()
        self._draw_graph_sample()

    def _draw_graph_sample(self):
        k = 40
        sampled_nodes = random.sample(self.network.nodes, k)
        sampled_graph = self.network.subgraph(sampled_nodes)

        in_degrees = list(self.network.in_degree(list(sampled_graph.nodes)))

        figure = plt.figure(figsize=(5, 5))
        network_canvas = FigureCanvas(figure)
        figure.clf()
        figure.suptitle('Sample subgraph (40 nodes)', fontsize=14)
        pos = nx.random_layout(sampled_graph)
        groups = set(nx.get_node_attributes(sampled_graph, 'class').values())
        mapping = dict(zip(sorted(groups), count()))
        nodes = sampled_graph.nodes()
        colors = [mapping[sampled_graph.nodes[n]['class']] for n in nodes]
        nx.draw(sampled_graph, pos=pos, with_labels=False, node_color=colors, nodelist=[n[0] for n in in_degrees], node_size=[n[1] for n in in_degrees])
        network_canvas.draw_idle()
        self.layout_graph.addWidget(network_canvas)

        self._calc_all_strength()

    def _calc_all_strength(self):
        _feature_strength = FeatureStrength(self.network)
        strengths_df = _feature_strength.compute_strenghts(self.feature_names)

        strengths_df.to_csv(f"{self.dataset_name.replace('.csv', '')}_{self.current_model}_strengths.csv", index=False)
        self.model = PandasModel(strengths_df)
        self.table_strength_single.setModel(self.model)

        as_df = _feature_strength.compute_avg_strenghts()
        self.model2 = PandasModel(as_df)
        self.table_strength_full.setModel(self.model2)
