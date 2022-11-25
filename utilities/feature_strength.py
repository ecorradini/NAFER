import networkx as nx
import math


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
            print(f"s_f{k}{vi[0]}_f{k}{vj[0]} {round((1 - pi) * pj * (1 - dist), 4)}")
            return round((1 - pi) * pj * (1 - dist), 4)
        else:
            print(f"s_f{k}{vi[0]}_f{k}{vj[0]} {round((1 - pi) * (1 - pj) * dist, 4)}")
            return round((1 - pi) * (1 - pj) * dist, 4)

    def _damping_factor(self, vi, k):
        v_out = []
        for edge in self.network.edges:
            if edge[0] == vi[0]:
                v_out.append((edge[1], self.network.nodes[edge[1]]))
        _strength_out = [self._s_fki_fkj(vi, (key, node), k) for key, node in v_out]
        return round(self._sig(sum(_strength_out) / len(_strength_out) if len(v_out) > 0 else 0, len(self.network.nodes)), 4)

    def _s_fki(self, vi, k):
        if f"strength_{k}" in vi[1]:
            return vi[1][f"strength_{k}"]

        _dki = self._damping_factor(vi, k)
        print(f"d{k}{vi[0]} {_dki}")

        # CASO DAMPING FACTOR 0 (no archi uscenti)
        if _dki == 0:
            _strength = 1 / len(self.network.nodes)
            nx.set_node_attributes(self.network, {vi[0]: {f"strength_{k}": _strength}})
            return _strength

        _left = (1 - _dki) / len(self.network.nodes)

        v_in = []
        for edge in self.network.edges:
            if edge[1] == vi[0]:
                v_in.append((edge[0], self.network.nodes[edge[0]]))

        # CASO NO ARCHI ENTRANTI
        if _dki > 0 and len(v_in) == 0:
            _strength = (1 - _dki) / len(self.network.nodes)
            nx.set_node_attributes(self.network, {vi[0]: {f"strength_{k}": _strength}})
            return _strength

        _right = _dki * sum([self._s_fki((key, node), k) / self.network.out_degree(key) for key, node in v_in])
        _strength = _left + _right
        nx.set_node_attributes(self.network, {vi[0]: {f"strength_{k}": _strength}})
        return _strength

    def get_instance_feature_strength(self, vi, k):
        return self._s_fki(vi, k)

