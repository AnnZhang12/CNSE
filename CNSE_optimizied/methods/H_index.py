import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def h_index_centrality(graph):
    """
    计算H指数中心性
    :param graph: 无向图，使用字典表示，键为节点，值为与该节点相连的节点列表
    :return: H指数中心性字典，键为节点，值为H指数中心性值
    """
    nodes = []
    for node in graph.nodes():
        nodes.append(node)
    # print(nodes)

    n = len(nodes)
    h_index = {node: 0 for node in nodes}
    for i, node in enumerate(nodes):
        degrees = [len(graph[n]) for n in graph[node]]
        degrees.sort(reverse=True)
        for j, degree in enumerate(degrees):
            if degree >= j + 1:
                h_index[node] = j + 1
            else:
                break
    # h_centrality = {node: h_index[node] / n for node in nodes} 归一化处理
    h_centrality = {node: h_index[node] for node in nodes}

    return h_centrality


# 扩展的h-index指数
def extended_h_index_centrality(graph):
    # print(graph.nodes())
    # 计算每个节点的 h-index
    h_index = {}
    for node in graph:
        citations = sorted([len(graph[n]) for n in graph[node]], reverse=True)
        for i, c in enumerate(citations):
            if i >= c:
                h_index[node] = i
                break
        else:
            h_index[node] = len(citations)

    # 计算每个节点的扩展 h-index 中心性
    centrality = {}
    for node in graph:
        extended_h_index = h_index[node]
        for neighbor in graph[node]:
            extended_h_index += h_index.get(neighbor, 0)
        centrality[node] = extended_h_index

    return centrality
