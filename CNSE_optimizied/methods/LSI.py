import networkx as nx
import matplotlib.pyplot as plt

def loadNetwork(file):
    graph = nx.read_edgelist(file)
    return graph


def degree1(graph, node):
    deg_1 = 0
    for nbr in list(graph.neighbors(node)):
        deg_1 = deg_1+graph.degree(nbr)
    return deg_1


def lsi_calculate(graph):
    lsi_dic = {}
    for node in graph.nodes():
        # print('!!!!!!!!!!!!!!!!!!!!!!', node)
        degree = graph.degree(node)
        deg1 = degree1(graph, node)
        lsi = (degree - (1 / degree * deg1)) / (degree + (1 / degree * deg1))
        # print(node, lsi)
        lsi_dic[node] = lsi
    return lsi_dic

def community_center(lsi_dic):
    Community_Center = {k for k, v in lsi_dic.items() if v >= 0}
    return Community_Center

# graph = loadNetwork('test1.txt')
# graph = create_Graph(f"/dataset/art_network1.cites")
# nx.draw(graph, with_labels=True, font_weight='bold')
# plt.show()
# LSI = lsi_calculate(graph)
# print(LSI)
# print(community_center(LSI), len(community_center(LSI)))

