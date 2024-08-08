from Query_GraphAdj import DirectedGraph
import numpy as np

G = DirectedGraph().get_G()
adjacency_matrix = DirectedGraph().get_adjacency_matrix(G)
edge_to_id = DirectedGraph().get_edge_id_order(G)
file3 = open("./EventData")
file4 = "./DynamicFeatureData"
subG_edges = set()

with open(file4, 'w') as file_object:
    while True:
        line_file3 = file3.readline()
        if not line_file3:
            break
        linelist = line_file3.split(',')
        if float(linelist[2]) < 0.5:
            continue
        int_way_id = int(linelist[1])
        if int_way_id not in edge_to_id.keys():
            continue
        subG_edges.add(int_way_id)
        file_object.write(line_file3)

subG_edges2 = set()
for node_id in subG_edges:
    node_neighbors = DirectedGraph().get_k_order_neighborhood(G, adjacency_matrix, node_id, 8)
    subG_edges2.update(node_neighbors)

merged_set = subG_edges.union(subG_edges2)
np.save('subG_edges.npy', list(merged_set))
