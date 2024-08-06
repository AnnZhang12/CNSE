from collections import Counter

import numpy as np

from NetworkLoader import NetworkLoader

from methods.CNSE_c_NLH_L_L_E import CNSE_C_NLH_LLE
from sklearn import metrics
import community
import pandas as pd
import networkx as nx

from methods.NewJaccard_3 import get_2_order_similarity
import time
from methods.ERM2 import get_ERM
from methods.LSI import lsi_calculate

from methods.NewJaccard_3 import get_2_order_similarity
from methods.H_index import h_index_centrality, extended_h_index_centrality
from methods.NGC_NEW import NGC_node, find_Community_Center_node, NGC_dic

epochs = 1


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
    return nmi, ari, predicted_label_list

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

    df = pd.DataFrame(columns=['Epoch', 'Method', 'NMI', 'ARI', 'NumofCommunity', "Best_params"])

    for epoch in range(epochs):
        # ================================RefineDCN=================================

        if best_params == -1:
            nmi_max = 0
            ari_max = 0
            predicted_label_list_final = []
            ip = calculate_importance_with_triangles(graph)
            lsi = lsi_calculate(graph)
            erm = get_ERM(graph)
            eh = extended_h_index_centrality(graph)
            ngc = NGC_dic(graph)
            ngc_nodes = NGC_node(graph)

            for n in range(int(0.2 * len(graph)), int(0.3 * len(graph))):

                for x in np.linspace(0, 1, 11):

                    for y in np.linspace(0, 1, 11):
                        z = 1 - x - y
                        if z > 0:  # 满足每个参数大于0的条件
                            print('xyz:', x, y, z)
                            rdcn_c_nlh = CNSE_C_NLH_LLE(graph, ip, lsi, erm, eh, ngc, ngc_nodes, n, x, y, z, second_order_jaccard_similarity)
                            predicted_label_dict = rdcn_c_nlh.detect_communities()
                            nmi, ari, predicted_label_list = evaluate(predicted_label_dict, node_community_label_list)
                            df.loc[len(df)] = [epoch, 'RefineDCN_C_NLH', nmi, ari, len(set(predicted_label_list)), (x, y, z)]

                            if nmi > nmi_max:
                                nmi_max = nmi
                                ari_max = ari
                                best_params = (x, y, z)
                                predicted_label_list_final = predicted_label_list
            df.loc[len(df)] = [epoch, 'BEST', nmi_max, ari_max, len(set(predicted_label_list_final)), best_params]

        else:
            rdcn_c_nlh = CNSE_C_NLH_LLE(graph, ip, lsi, erm, eh, ngc, ngc_nodes, n, x, y, z, second_order_jaccard_similarity)
            predicted_label_dict = rdcn_c_nlh.detect_communities()
            nmi, ari, predicted_label_list = evaluate(predicted_label_dict, node_community_label_list)
            df.loc[len(df)] = [epoch, 'BEST', nmi, ari, len(set(predicted_label_list)), best_params]



