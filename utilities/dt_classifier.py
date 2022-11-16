from sklearn.tree import DecisionTreeClassifier
import pandas as pd

class DTClassifier:
    def __init__(self, x_train, x_test, y_train, y_test):
        super(DTClassifier, self).__init__()
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def run(self):
        clf = DecisionTreeClassifier()
        clf = clf.fit(self.x_train, self.y_train)

        return clf.predict(self.x_test)
