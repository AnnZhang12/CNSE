import optuna
import numpy as np
from sklearn import metrics
import community


from methods.CNSE_c_NLH_L_L_E import CNSE_C_NLH_LLE

def evaluate(label_dict, node_community_label_list, graph):
    predicted_label_list = []
    true_label_list = []

    for key, value in label_dict.items():
        predicted_label_list.append(label_dict[key])
        true_label_list.append(node_community_label_list[key])

    nmi = metrics.normalized_mutual_info_score(true_label_list, predicted_label_list)
    return nmi

class EarlyStoppingCallback:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_value = float('inf')

    def __call__(self, study, trial):
        value = trial.value

        if value < self.best_value:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            study.stop()

def objective(trial):
    global g
    global second_j
    global train_epochs
    global true_label
    global ip1
    global lsi1
    global erm1
    global eh1
    global ngc1
    global ngc_nodes1

    param_n = trial.suggest_int('n', int(0.2 * len(g)), int(0.5 * len(g)))
    param_x = trial.suggest_float('x', 0, 1)
    param_y = trial.suggest_float('y', 0, 1 - param_x)
    param_z = 1 - param_x - param_y

    rdcn_c_nlh = CNSE_C_NLH_LLE(g, ip1, lsi1, erm1, eh1, ngc1, ngc_nodes1, param_n, param_x, param_y, param_z, second_j)
    predicted_label_dict = rdcn_c_nlh.detect_communities()
    nmi = evaluate(predicted_label_dict, true_label, g)
    return nmi


def rDCN_optimizer(graph,  ip, lsi, erm, eh, ngc, ngc_nodes, second_order_jaccard_similarity, node_community_label_list, epochs):
    global g, second_j, true_label, train_epochs, ip1, lsi1, erm1, eh1, ngc1, ngc_nodes1
    g = graph
    second_j = second_order_jaccard_similarity
    train_epochs = epochs
    true_label = node_community_label_list
    ip1 = ip
    lsi1 = lsi
    erm1 = erm
    eh1 = eh
    ngc1 = ngc
    ngc_nodes1 = ngc_nodes

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=1))  # sampler采样器方法

    study.optimize(objective, n_trials=1000)  # n_trials执行的试验次数

    trial = study.best_trial  # 最优超参对应的trial，有一些时间、超参、trial编号等信息；

    return trial.value, trial.params

