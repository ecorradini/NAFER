from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split

from utilities.mlp_classifier import MLPClassifier
from utilities.nb_classifier import NBClassifier
from utilities.rf_classifier import RFClassifier
from utilities.svmp_classifier import SVMPClassifier
from utilities.svmr_classifier import SVMRClassifier


class Classifier:
    def __init__(self, data, model, class_label):
        super(Classifier, self).__init__()
        self.data = data
        self.model = model
        self.class_label = class_label

    def _preprocess(self):
        le = preprocessing.LabelEncoder()
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        non_numerics = self.data.select_dtypes(exclude=numerics)
        for column in non_numerics:
            le.fit(non_numerics[column])
            self.data[column] = le.transform(non_numerics[column])
        self.data.fillna(0, inplace=True)

        scaler = preprocessing.MinMaxScaler()
        _columns = _features = [x for x in self.data.columns if x != self.class_label]
        self.data[_columns] = scaler.fit_transform(self.data[_columns])

    def run(self):
        self._preprocess()
        feature_cols = [f for f in self.data.columns if f != self.class_label]
        x = self.data[feature_cols]
        y = self.data[self.class_label]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

        classifier = None
        if self.model == "NB":
            classifier = NBClassifier(x_train, x_test, y_train, y_test)
        elif self.model == "SVMP":
            classifier = SVMPClassifier(x_train, x_test, y_train, y_test)
        elif self.model == "SVMR":
            classifier = SVMRClassifier(x_train, x_test, y_train, y_test)
        elif self.model == "MLP":
            classifier = MLPClassifier(x_train, x_test, y_train, y_test)
        elif self.model == "RF":
            classifier = RFClassifier(x_train, x_test, y_train, y_test)

        y_pred = classifier.run()
        accuracy = metrics.accuracy_score(y_test, y_pred)

        classified_data = x_test
        labels = y_pred

        proba = classifier.get_proba()
        probabilities = []
        for proba in proba:
            probabilities.append(round(max(proba), 2))

        classified_data["_class"] = labels
        classified_data["_confidence"] = probabilities
        feature_names = [x for x in classified_data.columns if x not in ["_class", "_confidence"]]

        return accuracy, classified_data, feature_names