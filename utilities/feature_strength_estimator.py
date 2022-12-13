import networkx as nx
import pandas as pd

from utilities.feature_strength import FeatureStrength

class FeatureStrengthEstimator:
    def __init__(self, complete_network, base_network):
        super(FeatureStrengthEstimator, self).__init__()
        self.complete_network = complete_network
        self.base_network = base_network

    def _s_fki(self, vi, k):
        if vi[0] not in self.base_network:
            _, strength = FeatureStrength(self.complete_network).s_fki(vi, k)
            return strength
        else:
            return nx.get_node_attributes(self.base_network, f"strength_{k}")[vi[0]]

    def compute_estimated_strenghts(self, feature_names):
        _estimated_strengths = {"node": []}
        for key, node in self.complete_network.nodes.items():
            _features = node["features"]
            _estimated_strengths["node"].append(key)
            for k, feature in enumerate(_features):
                _strength = self._s_fki((key, node), k)
                try:
                    _estimated_strengths[feature_names[k]].append(_strength)
                except:
                    _estimated_strengths[feature_names[k]] = [_strength]
        return pd.DataFrame.from_dict(_estimated_strengths)

