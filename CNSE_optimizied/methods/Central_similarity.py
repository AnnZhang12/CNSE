
# 计算节点之间的相似性
import math

import numpy as np


def calculate_similarity(node1, node2, graph):
    if node1 == node2:
        return 1  # 节点与自身的相似度为0
    aij = 1 if graph.has_edge(node1, node2) else 0
    di = graph.degree(node1)
    dj = graph.degree(node2)

    t1_neighbors_i = set(graph.neighbors(node1))
    t1_neighbors_j = set(graph.neighbors(node2))
    common_neighbors = t1_neighbors_i.intersection(t1_neighbors_j)

    similarity = aij / math.sqrt(di * dj) + (len(common_neighbors) / (2 * math.sqrt(di * dj)))
    return similarity

# 创建相似性矩阵
def get_similarity_matrix(G):
    num_nodes = len(G.nodes())
    similarity_matrix = np.zeros((num_nodes, num_nodes))

    for i, node1 in enumerate(G.nodes()):
        for j, node2 in enumerate(G.nodes()):
            similarity = calculate_similarity(node1, node2, G)
            similarity_matrix[i, j] = similarity

    # 输出相似性矩阵
    # print("相似性矩阵:")
    # print(similarity_matrix)
    return similarity_matrix
