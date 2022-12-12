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

from utilities.feature_strength import FeatureStrength
from utilities.mlp_classifier import MLPClassifier
from utilities.nb_classifier import NBClassifier
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

    def _preprocess(self):
        le = preprocessing.LabelEncoder()
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        non_numerics = self.data.select_dtypes(exclude=numerics)
        for column in non_numerics:
            le.fit(non_numerics[column])
            self.data[column] = le.transform(non_numerics[column])
        self.data.fillna(0, inplace=True)

        scaler = preprocessing.MinMaxScaler()
        _columns = _features = [x for x in self.data.columns if x != self.current_class]
        self.data[_columns] = scaler.fit_transform(self.data[_columns])

        print(self.data)

    def _run_classifier(self):
        self._preprocess()
        feature_cols = [f for f in self.data.columns if f != self.current_class]
        x = self.data[feature_cols]
        y = self.data[self.current_class]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

        classifier = None
        if self.current_model == "NB":
            classifier = NBClassifier(x_train, x_test, y_train, y_test)
        elif self.current_model == "SVMP":
            classifier = SVMPClassifier(x_train, x_test, y_train, y_test)
        elif self.current_model == "SVMR":
            classifier = SVMRClassifier(x_train, x_test, y_train, y_test)
        elif self.current_model == "MLP":
            classifier = MLPClassifier(x_train, x_test, y_train, y_test)
        elif self.current_model == "RF":
            classifier = RFClassifier(x_train, x_test, y_train, y_test)

        y_pred = classifier.run()
        self.label_accuracy.setText(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")

        self.classified_data = x_test #pd.concat([x_train, x_test])
        labels = y_pred #list(y_train) + list(y_pred)

        proba = classifier.get_proba()
        probabilities = []
        for proba in proba:
            probabilities.append(round(max(proba), 2))

        self.classified_data["_class"] = labels
        self.classified_data["_confidence"] = probabilities
        self.feature_names = [x for x in self.classified_data.columns if x not in ["_class", "_confidence"]]

        self._build_network()

    def _build_network(self):
        nodes_dict = {}
        for index, row in self.classified_data.iterrows():
            _class = row["_class"]
            _confidence = row["_confidence"]
            _features = [row[x] for x in self.classified_data.columns if x not in ["_class", "_confidence"]]
            self.network.add_node(index)
            nodes_dict[index] = {
                "features": _features,
                "class": _class,
                "confidence": _confidence
            }
        nx.set_node_attributes(self.network, nodes_dict)

        edges = set()
        for node in nodes_dict.keys():
            for node2 in nodes_dict.keys():
                if node != node2 and \
                        nodes_dict[node]["confidence"] <= nodes_dict[node2]["confidence"] and \
                        (node2, node) not in edges:
                    edges.add((node, node2))
        self.network.add_edges_from(edges)

        print(len(self.network.nodes))
        print(len(self.network.edges))

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
        _strengths = {}
        for key, node in self.network.nodes.items():
            _features = node["features"]
            for k, feature in enumerate(_features):
                _dki, _strength = _feature_strength.get_instance_feature_strength((key, node), k)
                try:
                    _strengths[self.feature_names[k]].append(_strength)
                except:
                    _strengths[self.feature_names[k]] = [_strength]
        print(_strengths)
        s_df = pd.DataFrame.from_dict(_strengths)
        s_df.to_csv(f"{self.dataset_name.replace('.csv', '')}_{self.current_model}_strengths.csv", index=False)
        self.model = PandasModel(s_df)
        self.table_strength_single.setModel(self.model)

        _avg_strengths = {"feature": [], "strength": []}
        for feature, values in _strengths.items():
            _avg_strengths["feature"].append(feature)
            _avg_strengths["strength"].append(sum(values)/len(values))
        as_df = pd.DataFrame.from_dict(_avg_strengths)
        self.model2 = PandasModel(as_df)
        self.table_strength_full.setModel(self.model2)