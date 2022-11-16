from PyQt6.QtWidgets import QWidget
from PyQt6 import uic
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
import pandas as pd

from utilities.dt_classifier import DTClassifier


class ResultView(QWidget):

    def __init__(self, data, name, parent=None):
        super(ResultView, self).__init__(parent)
        uic.loadUi('ui/result_view.ui', self)
        self.data = data
        self.classified_data = None

        self.combo_class.clear()
        self.combo_class.addItems(self.data.columns)
        self.current_class = self.data.columns[0]
        self.combo_class.currentTextChanged.connect(self._on_class_combobox_changed)

        self.models = ["DT", "NB", "SVMP", "SVMR", "ANN", "RF"]
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
        if self.current_model == "DT":
            classifier = DTClassifier(x_train, x_test, y_train, y_test)
        y_pred = classifier.run()
        self.label_accuracy.setText(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")

        self.classified_data = pd.concat([x_train, x_test])
        labels = list(y_train) + list(y_pred)
        print(labels)
        self.classified_data["_class"] = labels
        print(self.classified_data)


