import numpy as np

import networkx as nx





class CNSE_C_NLH_LLE():
    def __init__(self, graph, ip, lsi, erm, eh, ngc, ngc_nodes, N, weight_A, weight_B, weight_C, second_order_jaccard_similarity):
        self.graph = graph
        self.weight_A = weight_A
        self.weight_B = weight_B
        self.weight_C = weight_C
        self.second_order_jaccard_similarity = second_order_jaccard_similarity
        self.N = N  # Number of community centers
        self.C = []  # Community centers
        self.unlabeled_nodes = []  # Nodes without community center labels
        self.labels = {}  # Dictionary to store the multi-labels for each node
        self.common_neighbors = []
        self.final_labels = {}
        self.common_neighbors_union = []
        self.comprehensive_influence = {}
        # self.similarity_matrix = get_similarity_matrix(self.graph)
        self.IP = ip
        self.lsi = lsi
        self.erm = erm
        self.eh = eh
        self.ngc = ngc
        self.NGC_Node = ngc_nodes
        # print('ip:', self.IP)

    def calculate_weight(self, func_values):
        total_sum = sum(func_values.values())
        weights = {node: value / total_sum for node, value in func_values.items()}
        return weights

    def calculate_f(self, weights, func_weight):
        return sum(weight * func_weight for node, weight in weights.items())

    def Voting_Select_Center_Node(self):
        # 三种中心节点选择方法的数据
        # function_A = {2: 10, 4: 8, 3: 6, 1: 4, 5: 2}
        # function_B = {4: 100, 2: 95, 1: 90, 3: 85, 5: 80}
        # function_C = {2: 20, 3: 16, 4: 16, 5: 14, 1: 10}
        ngc_center = self.ngc
        # print('ngc_center:', ngc_center)
        # erm_center = self.erm
        # print('erm_center:', erm_center)
        lsi_center = self.lsi
        extended_h_index_center = self.eh
        # print('extended_h_index_center:', extended_h_index_center)

        # 投票占比
        # weight_A = 0.3
        # weight_B = 0.4
        # weight_C = 0.3

        # 计算节点权重
        weights_A = self.calculate_weight(ngc_center)
        weights_B = self.calculate_weight(lsi_center)
        weights_C = self.calculate_weight(extended_h_index_center)

        # 计算 F 值
        F_values = {}
        for node in ngc_center.keys():
            W_A = weights_A[node]
            W_B = weights_B[node]
            W_C = weights_C[node]
            F = self.weight_A * W_A + self.weight_B * W_B + self.weight_C * W_C
            F_values[node] = F

        # 按照 F 值进行排序，选择前N个节点作为中心节点
        sorted_F_values = sorted(F_values.items(), key=lambda x: x[1], reverse=True)
        self.comprehensive_influence = dict(sorted_F_values)
        # print('comprehensive_influence:', self.comprehensive_influence)
        self.C = [node for node, _ in sorted_F_values[:self.N]]

        # print("选出的中心节点：", self.C)
        return self.C, self.comprehensive_influence

    def find_nodes_connected_to_multiple_centers(self):
        # 找到同时连接到2个或2个以上中心节点的节点
        nodes_connected_to_multiple_centers = []
        for node in self.graph.nodes():
            neighbors = set(self.graph.neighbors(node))
            # 返回包含存在于集合 x 和集合 y 中的元素的集合：
            if len(neighbors.intersection(self.C)) >= 2:
                nodes_connected_to_multiple_centers.append(node)

        return nodes_connected_to_multiple_centers

    def initialize_multi_labels(self):
        for node in self.graph.nodes:
            self.labels[node] = np.zeros(self.N)
            if node in self.C:

                center_index = self.C.index(node)
                # print('center index:', center_index)
                self.labels[node][center_index] = 1
            else:
                self.unlabeled_nodes.append(node)
        return self.labels, self.unlabeled_nodes

    # def propagate_labels(self):
    #     # Implement the label propagation logic here
    #     # Check if the neighbors of each center only connect to that center
    #     # If true, propagate the center's label to those neighbors
    #     for center in self.C:
    #         neighbors = list(self.graph.neighbors(center))
    #         # print('===center, neighbors===', center, neighbors)
    #         # center_nodes_common_neighbors = self.center_nodes_neighbors()
    #         center_nodes_common_neighbors = self.find_nodes_connected_to_multiple_centers()
    #         # print('==center, neighbor_centers==', center, center_nodes_common_neighbors)
    #         # print('==center, neighbor_centers1==', center, center_nodes_common_neighbors1)
    #         for nei in neighbors:
    #             if (nei not in center_nodes_common_neighbors) and (nei not in self.C):
    #                 # print('(nei not in center_nodes_common_neighbors) and (nei not in self.C)', nei)
    #                 self.labels[nei] = self.labels[center]
    #                 if nei in self.unlabeled_nodes:
    #                     self.unlabeled_nodes.remove(nei)
    #                 # print('=============nei====self.labels[nei]=============', nei, self.labels[nei])
    #                 # print('直接连接到仅一个中心点的节点被分配与该中心点相同的标签', len(self.unlabeled_nodes), self.unlabeled_nodes)
    #     return self.labels, self.unlabeled_nodes

    # 初始化中心节点的邻居节点
    def propagate_labels(self):
        center_nodes_common_neighbors = self.find_nodes_connected_to_multiple_centers()
        # print('共同邻居-------------', center_nodes_common_neighbors)
        # for common_c_node in center_nodes_common_neighbors:
        #     max_value = None  # 初始化最大值为None
        #     max_element = None  # 初始化最大值对应的元素为None
        #     for c_node in self.C:
        #         sim = self.similarity_matrix[common_c_node][c_node]
        #         # print('common_c_node, c_node, sim:', common_c_node, c_node, sim)
        #         if max_value is None or sim > max_value:
        #             max_value = sim
        #             max_element = self.labels[c_node]
        #             # print('max:', max_element)
        #     self.labels[common_c_node] = max_element
        #     if common_c_node in self.unlabeled_nodes:
        #         self.unlabeled_nodes.remove(common_c_node)
        for center_node in self.C:
            neighbors = set(self.graph.neighbors(center_node))
            # print('center_node:', center_node)
            # print('neighbors--------------------------', len(neighbors), neighbors)
            # 遍历中心节点的邻居节点
            for neighbor in neighbors:
                if neighbor != center_node and neighbor not in center_nodes_common_neighbors:
                    # print('neighbor', neighbor)
                    # 获取中心节点的其他邻居节点
                    other_neighbors = neighbors - {neighbor}
                    # print('other_neighbors:', other_neighbors)
                    # 检查邻居节点与中心节点的其他邻居节点是否构成三角形
                    for other_neighbor in other_neighbors:
                        if self.graph.has_edge(neighbor, other_neighbor):
                            # print('yes!!!!', neighbor, other_neighbor)
                            # 赋予与中心节点相同的标签向量
                            self.labels[neighbor] = self.labels[center_node]
                            # print('labels[neighbor]:', self.labels[neighbor])
                            if neighbor in self.unlabeled_nodes:
                                self.unlabeled_nodes.remove(neighbor)
                            break
        # print('self.labels:', self.labels)
        # print('self.unlabeled_nodes:', self.unlabeled_nodes)
        return self.labels, self.unlabeled_nodes

    def calculate_importance(self, node):
        # importance = self.erm
        # importance = self.IP
        # print('ERM_importance', importance)
        importance = self.lsi
        importance_of_node = importance[node]
        # print('importance:', node, importance_of_node)
        return importance_of_node

    # def calculate_jaccard_similarity(self, node1, node2):
    #     # Implement a method to calculate the Jaccard similarity between two nodes
    #     # Jaccard similarity = (number of common neighbors) / (total number of neighbors)
    #     neighbors1 = set(self.graph.neighbors(node1))
    #     neighbors2 = set(self.graph.neighbors(node2))
    #     common_neighbors = len(neighbors1.intersection(neighbors2))
    #     total_neighbors = len(neighbors1.union(neighbors2))
    #     jaccard_sim = common_neighbors / total_neighbors
    #     return jaccard_sim

    def distance_between_node_and_center_node(self):
        # 计算每个节点与每个中心节点的距离
        distances_dict = {}
        for node in self.graph.nodes():
            distances = []
            for center_node in self.C:
                try:
                    shortest_path_length = nx.shortest_path_length(self.graph, source=center_node, target=node)
                except nx.NetworkXNoPath:
                    shortest_path_length = -1  # No path exists between nodes
                distances.append(shortest_path_length)
            distances_dict[node] = distances
        return distances_dict

    def importance_of_central_nodes(self, center_importance_vector):
        # top_n_importance = list(self.comprehensive_influence.values())[:self.N]
        # print('top_n_importance:', top_n_importance)
        distances_dict = self.distance_between_node_and_center_node()
        # print('distances_dict:',  distances_dict)
        # centrality = np.zeros(len(top_n_importance))  # 初始化中心性向量
        centrality = {}
        for n in self.graph.nodes:
            # 计算中心性
            centrality[n] = [x * (1 / y) if y != 0 else 0 for x, y in zip(center_importance_vector, distances_dict[n])]
        return centrality

    # # 计算每个节点的重要性指标
    # def calculate_importance_with_triangles(self):
    #     importance_dict = {}
    #     for node in self.graph.nodes():
    #         degree = self.graph.degree(node)
    #         # print('degree:', degree)
    #         triangles = nx.triangles(self.graph, node)
    #         # print('triangles:', triangles)
    #         t1_neighbors = set(self.graph.neighbors(node))
    #         second_order_neighbors = set()
    #
    #         for t1_neighbor in t1_neighbors:
    #             t1_neighbor_neighbors = set(self.graph.neighbors(t1_neighbor))
    #             second_order_neighbors.update(t1_neighbor_neighbors.difference([node]))
    #
    #         aij_zero_second_order_neighbors = [vj for vj in second_order_neighbors if
    #                                            not self.graph.has_edge(node, vj)]
    #
    #         t2_count = len(aij_zero_second_order_neighbors)
    #
    #         importance = degree + (triangles + t2_count) / 2
    #         importance_dict[node] = importance
    #     # 进行规范化
    #     importance_values = list(importance_dict.values())
    #     min_importance = min(importance_values)
    #     max_importance = max(importance_values)
    #
    #     normalized_importance_dict = {}
    #     for node, importance in importance_dict.items():
    #         normalized_importance = 0.5 + 0.5 * ((importance - min_importance) / (max_importance - min_importance))
    #         normalized_importance_dict[node] = normalized_importance
    #     # print('normalized_importance_dict', normalized_importance_dict)
    #     self.IP = normalized_importance_dict
    #     # print('ip---', self.IP)
    #
    #     return self.IP

    def label_assignment(self, node, centrality):
        labeled_neighbors = [neighbor for neighbor in self.graph.neighbors(node) if neighbor not in self.unlabeled_nodes]
        # print('节点i的标记邻居节点:', node, labeled_neighbors)
        if not labeled_neighbors:
            # print('-------------NGC_node--------------:', node, NGC_node(self.graph)[node])
            return self.labels[self.NGC_Node[node]]

        weighted_sum = np.zeros(self.N)
        # print('初始化weighted_sum', weighted_sum)
        for neighbor in labeled_neighbors:
            # jaccard_sim = self.calculate_jaccard_similarity(node, neighbor)
            jaccard_sim = self.second_order_jaccard_similarity[node][neighbor]
            # attribute_similarity = attribute_similarity_matrix[node][neighbor]
            # jaccard_sim = 1
            # print('jaccard_sim, self.labels[neighbor]:', jaccard_sim, self.labels[neighbor])
            weighted_sum += jaccard_sim * self.labels[neighbor]
        # print('最终weighted_sum==============:', weighted_sum)
        weighted_sum += centrality[node]
        # print('最终weighted_sum--------------:', weighted_sum)

        return weighted_sum

    # def common_neighbors_between_any_two_cednter_nodes(self):
    #     # Find the common neighbors for any two center nodes
    #     common_neighbors_sets = []
    #     # print('=======gongt ', self.C)
    #     for i in range(len(self.C)):
    #         for j in range(i + 1, len(self.C)):
    #             node_i = self.C[i]
    #             node_j = self.C[j]
    #             common_neighbors_sets.append(set(self.graph.neighbors(node_i)) & set(self.graph.neighbors(node_j)))
    #
    #     # Compute the union of common neighbors
    #     self.common_neighbors_union = set.union(*common_neighbors_sets)
    #     return self.common_neighbors_union

    # 计算一个节点的社团归属度。也即一个节点的邻居节点中有多少比例在这个社团中。
    def belonging_degree(self, node, communityLabel):
        neighbors = list(self.graph.neighbors(node))
        sum = 0
        for neighbor in neighbors:
            if neighbor in self.final_labels and self.final_labels[neighbor] == communityLabel:
                sum = sum + 1
        return sum / len(neighbors)

    def node_calibration(self):
        # Create a new dictionary with keys not present in list1
        common_neighbors_between_any_two_center = self.find_nodes_connected_to_multiple_centers()
        # common_node_degree = {}
        # for node in common_neighbors_between_any_two_center:
        #     # nodes.append(node)
        #     # common_node_degree[node] = self.graph.degree(node)
        #     common_node_degree[node] = self.erm
        #     # common_node_degree[node] = self.comprehensive_influence[node]
        #     # node_degree[node] = self.IP[node]
        common_node_degree_sorted = dict(sorted(self.lsi.items(), key=lambda x: x[1], reverse=True))
        # print('=======', common_node_degree_sorted)
        new_dic = {key: value for key, value in self.final_labels.items() if key not in common_neighbors_between_any_two_center}
        # print('节点校准--删除中心节点的共同邻居', len(new_dic), new_dic)
        for common_neighbor in common_node_degree_sorted:
            max_value = None  # 初始化最大值为None
            max_element = None  # 初始化最大值对应的元素为None
            for communityLabel in range(self.N):
                bd = self.belonging_degree(common_neighbor, communityLabel)
                if max_value is None or bd > max_value:
                    max_value = bd
                    max_element = communityLabel
            self.final_labels[common_neighbor] = max_element

        return self.final_labels


    # def calibration(self):
    #     # 初始化
    #     communities_before = []
    #     for k, v in self.final_labels.items():
    #         communities_before.append(v)
    #
    #     communities_before_count = {}
    #     for i in communities_before:
    #         communities_before_count[i] = communities_before_count.get(i, 0) + 1
    #     communities_list = list(communities_before_count.keys())
    #
    #     # nodes = []
    #     # node_degree = {}
    #     # 为了避免信息传递能力较低的社区在标定过程中影响结果，节点标签标定从度数较大的节点开始。
    #     # for node in self.graph.nodes:
    #         # nodes.append(node)
    #         # node_degree[node] = self.graph.degree(node)
    #         # node_degree[node] = self.erm
    #         # node_degree[node] = self.comprehensive_influence[node]
    #         # node_degree[node] = self.IP[node]
    #     node_degree_sorted = sorted(self.erm.items(), key=lambda x: x[1], reverse=True)

        ca_dic = {}
        for i in node_degree_sorted:
            # 计算lic
            # 找到i的邻居节点，再判断在哪个社团
            # neighbors = list(self.G.neighbors(i[0]))
            for c in communities_list:
                ca_dic[c] = self.belonging_degree(i[0], c)
            # ca值最大的社团标签
            ans = max(ca_dic, key=lambda x: ca_dic[x])
            # 如果社团校准之前的社团标签和ca值最大的社团标签不一致，则将ca值最大的标签赋给当前节点
            if ans != self.final_labels[i[0]]:
                # print('T!!!', i[0])
                self.final_labels[i[0]] = ans
        return self.final_labels

    def detect_communities(self):
        # ERM = get_ERM(self.graph)
        # ERM_desc = dict(sorted(ERM.items(), key=lambda item: item[1], reverse=True))
        # print('ERM:', ERM_desc)
        # n = len(self.graph.nodes())
        # attribute_similarity_matrix = self.calculate_similarity_matrix(content_matrix, adjajency_matrix)
        # print('unlabeled_nodes:', self.unlabeled_nodes)
        # Step 1: Select community centers
        # community_centers = self.select_community_centers()
        community_centers = self.Voting_Select_Center_Node()
        # print('community_centers:', community_centers)
        # self.calculate_importance_with_triangles()
        center_importance_vector = [self.IP[node] for node in self.C]

        # print('center_importance_vector:', center_importance_vector)

        # Step 2: Propagate labels from community centers
        self.initialize_multi_labels()
        # print('initialize_multi_labels:', self.labels)
        # print('unlabeled_nodes_ini:', len(self.unlabeled_nodes), self.unlabeled_nodes)
        self.propagate_labels()
        # print('propagate_neighbors_labels:', self.labels)
        # print('unlabeled_nodes_pro:', len(self.unlabeled_nodes), self.unlabeled_nodes)

        # Step 3: Sort the remaining unlabeled nodes by importance
        self.unlabeled_nodes = sorted(self.unlabeled_nodes, key=lambda node: self.calculate_importance(node), reverse=True)
        # print('after sorted:', self.unlabeled_nodes)

        # Step 4: Iteratively assign labels to the remaining nodes
        # centrality = self.importance_of_central_nodes(center_importance_vector)
        # # print('centrality:', centrality)
        centrality = self.importance_of_central_nodes(center_importance_vector)
        # print('centrality:', centrality)
        while len(self.unlabeled_nodes) > 0:
            node = self.unlabeled_nodes.pop(0)
            # print('选择具有最高重要性的未标记节点来分配计算的标签', node)
            # self.labels[node] = self.label_assignment(node, attribute_similarity_matrix)

            self.labels[node] = self.label_assignment(node, centrality)
            # self.normalize_multi_label(node)

        # Step 5: Select the final label for each node based on the maximum proportion
        # final_labels = {}
        for node in self.graph.nodes:
            if node not in self.C:
                max_label = max(self.labels[node])
                max_index = np.where(self.labels[node] == max_label)[0][0]
                self.final_labels[node] = max_index
            else:
                center_index = self.C.index(node)
                self.final_labels[node] = center_index

        # print('归一化', self.labels)
        # print('final_labels:', len(self.final_labels), self.final_labels)
        self.node_calibration()
        # self.boundary_node_inspection()
        # self.calibration()

        return self.final_labels

# Example usage:
# Replace the graph variable and implement the missing functions based on your requirements.
# Then, you can run the community detection algorithm as follows:
# graph = YourGraphImplementation()  # Replace with your graph data
# cd = CommunityDetection(graph)
# results = cd.detect_communities()
# print(results)
