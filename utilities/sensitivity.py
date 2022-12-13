import pandas as pd

from utilities.classifier import Classifier
from utilities.feature_strength import FeatureStrength
from utilities.feature_strength_estimator import FeatureStrengthEstimator
from utilities.network import Network


class Sensitivity:
    def __init__(self, data, model, class_label, percentage):
        super(Sensitivity, self).__init__()
        self.data = data
        self.model = model
        self.class_label = class_label
        self.percentage = percentage

    def compute(self):
        self._run_classifier()
        self._split_data()
        self._build_base_network()
        self._build_complete_network()
        return self._estimate_strenghts()

    def _run_classifier(self):
        classifier = Classifier(self.data, self.model, self.class_label)
        _, self.classified_data, self.feature_names = classifier.run()

    def _split_data(self):
        self.test = self.classified_data.sample(n=int((len(self.classified_data)*self.percentage)/100))
        f = pd.concat([self.classified_data, self.test])
        self.base = f.drop_duplicates(keep=False)

    def _build_base_network(self):
        self.base_network = Network(self.base).build_network()
        FeatureStrength(self.base_network).compute_strenghts(self.feature_names)

    def _build_complete_network(self):
        self.complete_network = Network(self.classified_data).build_network()
        self.real_strengths = FeatureStrength(self.complete_network).compute_strenghts(self.feature_names)
        print(self.real_strengths)

    def _estimate_strenghts(self):
        estimated_strengths = FeatureStrengthEstimator(self.complete_network, self.base_network)\
            .compute_estimated_strenghts(self.feature_names)
        print(estimated_strengths)
        self.sensitivities = (self.real_strengths.drop("node", axis=1) - estimated_strengths.drop("node", axis=1)).abs()
        self.sensitivities["node"] = estimated_strengths["node"]
        return self.sensitivities

    def compute_overall_sensitivity(self):
        df = self.sensitivities.drop("node", axis=1).mean().to_frame()
        df.columns = ["sensitivity"]
        return df


