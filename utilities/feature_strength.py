import networkx as nx
import scipy


class FeatureStrength:
    def __init__(self, network):
        super(FeatureStrength, self).__init__()
        self.network = network

    def _s_fki_fkj(self, vi, vj, k):
        ag = vi[1]["confidence"] if vi[1]["class"] != vj[1]["class"] else vi[1]["confidence"]*vj[1]["confidence"]
        dist = abs(vi[1]["features"][k] - vj[1]["features"][k])
        return ag*dist

    def _damping_factor(self, vi, k):
        v_in = []
        v_out = []
        for edge in self.network.edges:
            if edge[1] == vi[0]:
                v_in.append((edge[0], self.network.nodes[edge[0]]))
            elif edge[0] == vi[0]:
                v_out.append((edge[1], self.network.nodes[edge[1]]))
        #_left = self.network.in_degree(vi[0])/self.network.out_degree(vi[0]) if self.network.out_degree(vi[0]) > 0 else 0
        _strength_in = [self._s_fki_fkj(vi, (key, node), k)/len(v_in) for key, node in v_in]
        _strength_out = [self._s_fki_fkj(vi, (key, node), k)/len(v_out) for key, node in v_out]
        return v_in, 0.85 #sum(_strength_in)/len(v_in) if len(v_in) > 0 else 0
        #_sum_out = (sum(_strength_out)-min(_strength_out))/(max(_strength_out)-min(_strength_out))
        #return v_in, abs((_sum_out/len(v_out) if len(v_out) > 0 else 1)) - abs((_sum_in/len(v_in) if len(v_in) > 0 else 0))
        #return v_in, abs((_sum_in / len(v_in) if len(v_in) > 0 else 1) * (len(v_out) / _sum_out if len(v_out) > 0 else 1))
        #return v_in, abs((len(v_out)/(len(v_in)+len(v_out))) * _sum_out - (len(v_in)/(len(v_in)+len(v_out)))*_sum_in)

    def _s_fki(self, vi, k):
        if f"strength_{k}" in vi[1]:
            return vi[1][f"strength_{k}"]
        all_other_nodes, _dki = self._damping_factor(vi, k)
        _left = (1 - _dki) / len(self.network.nodes)
        print(f"d{k}{vi[0]} = {_dki}")
        if self.network.out_degree(vi[1]) == 0:
            _strength = _left + _dki
            nx.set_node_attributes(self.network, {vi[0]: {f"strength_{k}": _strength}})
            return _strength
        _right = _dki*sum([(self._s_fki((key, node), k)/self.network.out_degree(key) if self.network.out_degree(key) > 0 else 0) for key, node in all_other_nodes])
        _strength = _left + _right
        if f"strength_{k}" not in vi[1]:
            nx.set_node_attributes(self.network, {vi[0]: {f"strength_{k}": _strength}})
        return _strength

    def get_instance_feature_strength(self, vi, k):
        return self._s_fki(vi, k)

