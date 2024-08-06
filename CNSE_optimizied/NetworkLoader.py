import glob
import numpy as np


class NetworkLoader():

    def __init__(self, data):
        self.dataDir = 'data/' + data + '/'
        self.fileName = data


    def network_parser(self, path):
        node_count_dic = {}
        counter = 0
        att_list = []
        node_community_label_list = []

        f1 = self.dataDir + self.fileName + '.content'
        with open(f1) as f:
            while True:
                line = f.readline()
                if line == '':
                    break
                tmp = line.split("\t")
                node_count_dic[tmp[0]] = counter
                # print('content:', tmp, counter)
                counter += 1
                att_list.append((tmp[1:-1]))
                node_community_label_list.append(tmp[-1].replace("\n", ""))

        edge_list = []
        f2 = self.dataDir + self.fileName + '.cites'
        with open(f2) as f:
            while True:
                line = f.readline()
                if line == '':
                    break
                line = line.replace("\n", "")
                tmp = line.split()
                if tmp[0] in node_count_dic and tmp[1] in node_count_dic:
                    ind0 = node_count_dic[tmp[0]]
                    ind1 = node_count_dic[tmp[1]]
                    edge_list.append((ind0, ind1))

        number_of_node = len(node_count_dic)
        att_size = len(att_list[0])
        adjajency_matrix = np.zeros((number_of_node, number_of_node))
        content_matrix = np.zeros((number_of_node, att_size))

        for i in range(len(edge_list)):
            adjajency_matrix[edge_list[i][0], edge_list[i][1]] = 1
            adjajency_matrix[edge_list[i][1], edge_list[i][0]] = 1

        for i in range(len(att_list)):
            for j in range(len(att_list[0])):
                content_matrix[i, j] = float(att_list[i][j])

        return adjajency_matrix, content_matrix, node_community_label_list, edge_list  # scaling S and A







