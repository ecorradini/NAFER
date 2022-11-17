from itertools import count

from PyQt6.QtWidgets import QWidget
from PyQt6 import uic
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import random

from utilities.nb_classifier import NBClassifier


class ResultView(QWidget):

    def __init__(self, data, name, parent=None):
        super(ResultView, self).__init__(parent)
        uic.loadUi('ui/result_view.ui', self)
        self.data = data
        self.classified_data = None
        self.network = nx.DiGraph()

        self.combo_class.clear()
        self.combo_class.addItems(self.data.columns)
        self.current_class = self.data.columns[0]
        self.combo_class.currentTextChanged.connect(self._on_class_combobox_changed)

        self.models = ["NB", "SVMP", "SVMR", "ANN", "RF"]
        self.combo_models.clear()
        self.combo_models.addItems(self.models)
        self.combo_models.currentTextChanged.connect(self._on_model_combobox_changed)
        self.current_model = self.models[0]

        self.button_run.clicked.connect(self._run_classifier)

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

    def _run_classifier(self):
        self._preprocess()
        feature_cols = [f for f in self.data.columns if f != self.current_class]
        x = self.data[feature_cols]
        y = self.data[self.current_class]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

        classifier = None
        if self.current_model == "NB":
            classifier = NBClassifier(x_train, x_test, y_train, y_test)
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

        self.build_network()

    def build_network(self):
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

        self.draw_graph_sample()

    def draw_graph_sample(self):
        k = 20
        sampled_nodes = random.sample(self.network.nodes, k)
        sampled_graph = self.network.subgraph(sampled_nodes)

        figure = plt.figure(figsize=(4, 4))
        network_canvas = FigureCanvas(figure)
        figure.clf()
        figure.suptitle('Sample subgraph (20 nodes)', fontsize=14)
        pos = nx.random_layout(sampled_graph)
        groups = set(nx.get_node_attributes(sampled_graph, 'class').values())
        mapping = dict(zip(sorted(groups), count()))
        nodes = sampled_graph.nodes()
        colors = [mapping[sampled_graph.nodes[n]['class']] for n in nodes]
        nx.draw(sampled_graph, pos=pos, with_labels=False, node_color=colors)
        network_canvas.draw_idle()
        self.layout_graph.addWidget(network_canvas)







