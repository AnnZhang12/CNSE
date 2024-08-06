# 作者：Ann

import math

import networkx as nx
import matplotlib.pyplot as plt
def set_Entropy_Centrality(G):
    for node in G.nodes():
        G.add_node(node, Entropy_Centrality=float, deg_1=int, deg_2=int)
    for node in G.nodes():
        # 计算一阶度
        deg_1 = 0
        for nbr in list(G.neighbors(node)):
            deg_1 = deg_1+G.degree(nbr)
        G.nodes[node]["deg_1"] = deg_1
        #print(G.nodes[node]["deg_1"])
    deg2 = []
    for node in G.nodes():
        # 计算二阶度
        deg_2 = 0
        for nbr in list(G.neighbors(node)):
            deg_2 = deg_2+G.nodes[nbr]["deg_1"]
        G.nodes[node]["deg_2"] = deg_2
        deg2.append(deg_2)
        # print(G.nodes[node])
        # print(type(G.nodes[node]))

    # 所有节点中二阶度的最大值
    max_deg = max(deg2)
    # print('max_deg', max_deg)
    for node in G.nodes():
        # 计算熵
        # E1
        E = 0
        for nbr in list(G.neighbors(node)):
            E = E + (G.degree(nbr)/G.nodes[node]["deg_1"])*math.log((G.degree(nbr)/G.nodes[node]["deg_1"]), 2)
        E1 = -E
        # E2
        E = 0
        for nbr in list(G.neighbors(node)):
            E = E + (G.nodes[nbr]["deg_1"] / G.nodes[node]["deg_2"]) * math.log((G.nodes[nbr]["deg_1"] / G.nodes[node]["deg_2"]),2)
        E2 = -E
        lam = G.nodes[node]["deg_2"] / max_deg
        G.nodes[node]["Entropy_Centrality"] = E1+lam*E2

def get_ERM(G):
    set_Entropy_Centrality(G)
    result = {}
    for node in G.nodes():
        SI = 0
        for nbr in list(G.neighbors(node)):
            for nbr2 in list(G.neighbors(nbr)):
                SI = SI + G.nodes[nbr2]["Entropy_Centrality"]
        result[node] = SI
    return result