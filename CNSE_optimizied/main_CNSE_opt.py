from collections import Counter

import numpy as np

from NetworkLoader import NetworkLoader
from sklearn import metrics
import community
import pandas as pd
import networkx as nx

from methods.NewJaccard_3 import get_2_order_similarity
from CNSE_optimizer_NMI import rDCN_optimizer

import time
from methods.ERM2 import get_ERM
from methods.LSI import lsi_calculate

from methods.NewJaccard_3 import get_2_order_similarity
from methods.H_index import h_index_centrality, extended_h_index_centrality
from methods.NGC_NEW import NGC_node, find_Community_Center_node, NGC_dic

epochs = 0


def calculate_2_order_jaccard_similarity(graph):
    A = np.array(nx.adjacency_matrix(graph).todense())
    second_order_jaccard_similarity = get_2_order_similarity(A)
    return second_order_jaccard_similarity

def evaluate(label_dict, node_community_label_list):
    predicted_label_list = []
    true_label_list = []

    for key, value in label_dict.items():
        predicted_label_list.append(label_dict[key])
        true_label_list.append(node_community_label_list[key])

    nmi = metrics.normalized_mutual_info_score(true_label_list, predicted_label_list)
    modularity = community.modularity(label_dict, graph)
    ari = metrics.adjusted_rand_score(true_label_list, predicted_label_list)
    return nmi, modularity, ari, predicted_label_list

def calculate_importance_with_triangles(graph):
    importance_dict = {}
    for node in graph.nodes():
        degree = graph.degree(node)
        triangles = nx.triangles(graph, node)
        t1_neighbors = set(graph.neighbors(node))
        second_order_neighbors = set()

        for t1_neighbor in t1_neighbors:
            t1_neighbor_neighbors = set(graph.neighbors(t1_neighbor))
            second_order_neighbors.update(t1_neighbor_neighbors.difference([node]))

        aij_zero_second_order_neighbors = [vj for vj in second_order_neighbors if
                                           not graph.has_edge(node, vj)]

        t2_count = len(aij_zero_second_order_neighbors)

        importance = degree + (triangles + t2_count) / 2
        importance_dict[node] = importance
    # 进行规范化
    importance_values = list(importance_dict.values())
    min_importance = min(importance_values)
    max_importance = max(importance_values)

    normalized_importance_dict = {}
    for node, importance in importance_dict.items():
        normalized_importance = 0.5 + 0.5 * ((importance - min_importance) / (max_importance - min_importance))
        normalized_importance_dict[node] = normalized_importance
    IP = normalized_importance_dict


    return IP

def NGC_dic(G):
    ngc_nodes = NGC_node(G)

    value_counts = Counter(ngc_nodes.values())
    all_keys = range(len(G))  # key属于[0, 60]
    completed_dict = {key: value_counts.get(key, 0) for key in all_keys}

    return completed_dict


datasets = ['LFR1']

for item in datasets:
    best_params = -1
    loader = NetworkLoader(item)
    adjajency_matrix, content_matrix, node_community_label_list, edge_list = loader.network_parser('data/' + item)
    graph = nx.Graph(edge_list)

    second_order_jaccard_similarity = calculate_2_order_jaccard_similarity(graph)
    ip = calculate_importance_with_triangles(graph)
    lsi = lsi_calculate(graph)
    erm = get_ERM(graph)
    eh = extended_h_index_centrality(graph)
    ngc = NGC_dic(graph)
    ngc_nodes = NGC_node(graph)

    df = pd.DataFrame(columns=['datasets', 'Epoch', 'Method', 'NMI', "Best_params"])

    NMI, x = rDCN_optimizer(graph,  ip, lsi, erm, eh, ngc, ngc_nodes, second_order_jaccard_similarity, node_community_label_list, epochs)

    df.loc[len(df)] = [item, epochs, 'RefineDCN_C', NMI, x]




