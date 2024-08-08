import networkx as nx
import numpy as np

class DirectedGraph:
    def __init__(self):
        with open('./TrafficDataset/edges', 'r') as file:
            lines = file.readlines()
            data = [tuple(map(int, line.strip().split(','))) for line in lines]
        self.G = nx.DiGraph()
        for start, end in data:
            self.G.add_edge(start, end)

    def get_G(self):
        return self.G

    def get_adjacency_matrix(self, G):
        return nx.adjacency_matrix(G)

    def get_edge_id_order(self, G):
        nodes = list(G.nodes())
        node2id = {nodes[i]: i for i in range(len(nodes))}
        return node2id

    def get_k_order_neighborhood(self, G, adjacency_matrix, node_id, k):
        edg2id = self.get_edge_id_order(G)
        id2edg = {v: k for k, v in edg2id.items()}
        edge_id = edg2id[node_id]

        if edge_id < 0 or edge_id >= adjacency_matrix.shape[0]:
            raise ValueError("Invalid edge_id")
        if k < 0:
            raise ValueError("k must be a non-negative integer")

        current_order = 0
        neighborhood_set = {edge_id}

        while current_order < k:
            current_order += 1
            current_neighbors = set()

            for node in neighborhood_set:
                neighbors = np.nonzero(adjacency_matrix[node, :])[1]
                current_neighbors.update(neighbors)
            neighborhood_set.update(current_neighbors)

        neighborhood_id_set = {id2edg[edge_num] for edge_num in neighborhood_set}
        return list(neighborhood_id_set)

    def sub_graph(self, sub_nodes):
        G = self.G
        SubG = G.subgraph(sub_nodes)
        return SubG