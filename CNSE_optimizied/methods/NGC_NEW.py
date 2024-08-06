from collections import Counter

import numpy as np

import networkx as nx
from NetworkLoader import NetworkLoader
import matplotlib.pyplot as plt
from methods.ERM2 import get_ERM

from methods import H_index


def create_Graph(path):
    G = nx.Graph()
    Data = np.loadtxt(path)
    List_A = []
    List_B = []
    for row in range(Data.shape[0]):
        List_A.append(Data[row][0])
        List_B.append(Data[row][1])
    List_A = list(set(List_A))
    List_B = list(set(List_B))
    length_A = len(List_A)
    length_B = len(List_B)
    totalNodeNum = int(max(max(List_A), max(List_B)))
    print('    节点数量为：' + str(totalNodeNum))
    G.add_nodes_from([i for i in range(totalNodeNum)])
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            strlist = line.split()
            n1 = int(strlist[0]) - 1
            n2 = int(strlist[1]) - 1
            G.add_edges_from([(n1, n2)])
    return G

def NGC_node(G):
    # 计算每个节点的度中心性
    # degree_centralities = {0:49, 1:28, 2:40, 3:46, 4:29, 5:54, 6:32, 7:46, 8:14, 9:46, 10:28, 11:14}
    degree_centralities = H_index.extended_h_index_centrality(G)
    # degree_centralities = get_ERM(G)
    # print('degree_centralities:', degree_centralities)
    # 找到每个节点的NGC节点
    ngc_nodes = {}

    for node in G.nodes():
        # 获取一阶邻居节点
        neighbors = set(nx.neighbors(G, node))
        ngc_node = None

        # 遍历一阶邻居节点
        for neighbor in neighbors:
            if degree_centralities[neighbor] > degree_centralities[node]:
                # 如果存在度中心性更大的一阶邻居节点，将其作为NGC节点
                if ngc_node is None or degree_centralities[neighbor] > degree_centralities[ngc_node]:
                    ngc_node = neighbor

        if ngc_node is None:
            # 如果一阶邻居节点中不存在度中心性更大的节点，则在二阶邻居内寻找
            second_neighbors = set()

            for neighbor in neighbors:
                second_neighbors.update(nx.neighbors(G, neighbor))

            second_neighbors -= neighbors | {node}

            for second_neighbor in second_neighbors:
                if degree_centralities[second_neighbor] > degree_centralities[node]:
                    # 如果存在度中心性更大的二阶邻居节点，将其作为NGC节点
                    if ngc_node is None or degree_centralities[second_neighbor] > degree_centralities[ngc_node]:
                        ngc_node = second_neighbor

        if ngc_node is None:
            # 如果在一阶和二阶邻居中都没有度中心性更大的节点，将该节点本身作为NGC节点
            ngc_node = node

        # 将节点和其对应的NGC节点添加到字典中
        ngc_nodes[node] = ngc_node

    # 打印每个节点的NGC节点
    # for node, ngc_node in ngc_nodes.items():
    #     print(f"Node {node}: NGC Node {ngc_node}")
    return ngc_nodes


def NGC_dic(G):
    ngc_nodes = NGC_node(G)
    # 给定的字典
    # print('ngc_nodes:', ngc_nodes)

    # 使用计数器统计字典值的频次
    value_counts = Counter(ngc_nodes.values())
    # print('value_counts：', dict(value_counts))
    all_keys = range(len(G))  # key属于[0, 60]
    # print(all_keys)
    completed_dict = {key: value_counts.get(key, 0) for key in all_keys}

    # print("补全后的字典：", completed_dict)

    # # 按照频次从大到小排序值
    # sorted_values = sorted(value_counts, key=lambda x: value_counts[x], reverse=True)
    # print('NGC_dic:', sorted_values)

    # 取频次最大的两个值
    # top_K_values = sorted_values[:topk]
    # print('top_K_values:', top_K_values)
    # 打印频次最大的两个值
    # for value in top_two_values:
    #     print(f"Value: {value}, Frequency: {value_counts[value]}")
    return completed_dict

def find_Community_Center_node(G, topk):
    ngc_nodes = NGC_node(G)
    # 给定的字典
    # print('ngc_nodes:', ngc_nodes)

    # 使用计数器统计字典值的频次
    value_counts = Counter(ngc_nodes.values())

    # 按照频次从大到小排序值
    sorted_values = sorted(value_counts, key=lambda x: value_counts[x], reverse=True)
    # print('NGC_dic:', sorted_values)

    # 取频次最大的两个值
    top_K_values = sorted_values[:topk]
    # print('top_K_values:', top_K_values)
    # 打印频次最大的两个值
    # for value in top_two_values:
    #     print(f"Value: {value}, Frequency: {value_counts[value]}")
    return top_K_values


# G = create_Graph('G:\\ZNN\\LPAZNN_2023\\LPAZNN\\data\\ngc_sample.txt')
# nx.draw(G, with_labels=True)
# plt.show()
# Community_Center = find_Community_Center_node(G)