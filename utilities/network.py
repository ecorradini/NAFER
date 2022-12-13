import networkx as nx

class Network:
    def __init__(self, data):
        super(Network, self).__init__()
        self.data = data

    def build_network(self):
        network = nx.DiGraph()
        nodes_dict = {}
        for index, row in self.data.iterrows():
            _class = row["_class"]
            _confidence = row["_confidence"]
            _features = [row[x] for x in self.data.columns if x not in ["_class", "_confidence"]]
            network.add_node(index)
            nodes_dict[index] = {
                "features": _features,
                "class": _class,
                "confidence": _confidence
            }
        nx.set_node_attributes(network, nodes_dict)

        edges = set()
        for node in nodes_dict.keys():
            for node2 in nodes_dict.keys():
                if node != node2 and \
                        nodes_dict[node]["confidence"] <= nodes_dict[node2]["confidence"] and \
                        (node2, node) not in edges:
                    edges.add((node, node2))
        network.add_edges_from(edges)

        return network