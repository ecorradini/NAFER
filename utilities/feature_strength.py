import networkx as nx
import math
import pandas as pd

class FeatureStrength:
    def __init__(self, network):
        super(FeatureStrength, self).__init__()
        self.network = network

    def _sig(self, x, V):
        return 1 / (1 + math.exp(-1*V*x))

    def _s_fki_fkj(self, vi, vj, k):
        dist = abs(vi[1]["features"][k] - vj[1]["features"][k])
        pi = vi[1]["confidence"]
        pj = vj[1]["confidence"]
        if vi[1]["class"] != vj[1]["class"]:
            #print(f"s_f{k}{vi[0]}_f{k}{vj[0]} {round((1 - pi) * pj * (1 - dist), 4)}")
            return round((1 - pi) * pj * (1 - dist), 4)
        else:
            #print(f"s_f{k}{vi[0]}_f{k}{vj[0]} {round((1 - pi) * (1 - pj) * dist, 4)}")
            return round((1 - pi) * (1 - pj) * dist, 4)

    def _damping_factor(self, vi, k):
        v_out = []
        for edge in self.network.edges:
            if edge[0] == vi[0]:
                v_out.append((edge[1], self.network.nodes[edge[1]]))
        _strength_out = [self._s_fki_fkj(vi, (key, node), k) for key, node in v_out]
        return round(self._sig(sum(_strength_out) / len(_strength_out) if len(v_out) > 0 else 0, len(self.network.nodes)), 4)

    def s_fki(self, vi, k):
        if f"strength_{k}" in vi[1]:
            return vi[1][f"d_{k}"], vi[1][f"strength_{k}"]

        _dki = self._damping_factor(vi, k)

        # CASO DAMPING FACTOR 0 (no archi uscenti)
        if _dki == 0:
            _strength = 1 / len(self.network.nodes)
            nx.set_node_attributes(self.network, {vi[0]: {f"strength_{k}": _strength}})
            nx.set_node_attributes(self.network, {vi[0]: {f"d_{k}": _dki}})
            return 0, _strength

        _left = (1 - _dki) / len(self.network.nodes)

        v_in = []
        for edge in self.network.edges:
            if edge[1] == vi[0]:
                v_in.append((edge[0], self.network.nodes[edge[0]]))

        # CASO NO ARCHI ENTRANTI
        if _dki > 0 and len(v_in) == 0:
            _strength = (1 - _dki) / len(self.network.nodes)
            nx.set_node_attributes(self.network, {vi[0]: {f"strength_{k}": _strength}})
            nx.set_node_attributes(self.network, {vi[0]: {f"d_{k}": _dki}})
            return _dki, _strength

        _right = _dki * sum([self.s_fki((key, node), k)[1] / self.network.out_degree(key) for key, node in v_in])
        _strength = _left + _right
        nx.set_node_attributes(self.network, {vi[0]: {f"strength_{k}": _strength}})
        nx.set_node_attributes(self.network, {vi[0]: {f"d_{k}": _dki}})
        return _dki, _strength

    def compute_strenghts(self, feature_names):
        self._strengths = {"node": []}
        for key, node in self.network.nodes.items():
            _features = node["features"]
            self._strengths["node"].append(key)
            for k, feature in enumerate(_features):
                _dki, _strength = self.s_fki((key, node), k)
                try:
                    self._strengths[feature_names[k]].append(_strength)
                except:
                    self._strengths[feature_names[k]] = [_strength]

        return pd.DataFrame.from_dict(self._strengths)

    def compute_avg_strenghts(self):
        _avg_strengths = {"feature": [], "strength": []}
        for feature, values in self._strengths.items():
            if feature != "node":
                _avg_strengths["feature"].append(feature)
                _avg_strengths["strength"].append(sum(values) / len(values))
        return pd.DataFrame.from_dict(_avg_strengths)

