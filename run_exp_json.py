import os
import numpy as np
from copy import deepcopy
from scipy.io import savemat, loadmat
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from util.FL_util import *
from util.read_db import get_dataloaders
from util.create_model import load_model
from util.clustering import get_gradients
from util.util_AoI import channel_model, get_c_vec_stat, generate_frequency, sample_c_vec
from util.util_lagrangian import per_slot_problem_Deploy_Gurobi_linear, get_lagrangian_infinite_mix, per_slot_problem_Deploy_DivFL,per_slot_problem_Deploy_Gurobi_linearSCA
import scipy.linalg as nla


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import json
json_file = sys.argv[1] 


with open(f"{json_file}.json") as f:
    param = json.load(f)
    
print(f"Using configuration file: {json_file}.")
print(param)

USE_WANDB = param["USE_WANDB"]
if USE_WANDB:
    WB_PROJECT = param["WB_PROJECT"]
OPTIMIZER = param["OPTIMIZER"]
DATASET = param["DATASET"]
AOI = param["AOI"]
INIT_METHOD = param["INIT_METHOD"]
FACTOR = param["FACTOR"]
LAMBDAA = param["LAMBDAA"]
ETA = param["ETA"]
N_CLIENT = param["N_CLIENT"]
M_SELECTED = param["M_SELECTED"]
RANDOM_SEED = param["RANDOM_SEED"]
C_BAR = param["C_BAR"]
A_MAX = param["A_MAX"]
KAPPA = param["KAPPA"]
N_SGD = param["N_SGD"]
LEARNING_RATE = param["LEARNING_RATE"]
DECAY = param["DECAY"]
MU = param["MU"] 
BATCH_SIZE = param["BATCH_SIZE"] 
ALGORITHM = param["ALGORITHM"]



# Selective import
if ALGORITHM == 'CS':
    from util.clustering import get_matrix_similarity_from_grads
    from util.clustering import get_clusters_with_alg2
    from util.clustering import sample_clients, get_gradients



if USE_WANDB:
    import wandb
    
if DATASET[:5] == "MNIST":
    N_ITER = param["MNIST_ITER"]    # maximum number of iterations
else:
    N_ITER = param["CIFAR10_ITER"]  #
BALANCED = DATASET[6:10] == "bbal"  # 

config={
    "eta": ETA,                     # Panelty factor used for evaluation
    "N": N_CLIENT,                  # Number of total users
    "A_max": A_MAX,                 # Max AoI considered
    "c_bar": C_BAR,                 # Discretized version of channels
    "num_action": 2,                # Number of actons per user
    "M": M_SELECTED,                # Number of sampled users
    "random_seed": RANDOM_SEED,
    "T": 200, 
    "kappa": 0.2,
    "lambda": LAMBDAA,
    "USE_AOI": 0, 
}
if AOI: 
    config["USE_AOI"] = 1

init_config = {
    "cost_matrix": channel_model(config),
    "c_vec_stat": get_c_vec_stat(config),
    "weight_vec": np.ones((config["N"],)),
    "mean_distribution": generate_frequency(config)
}

if ~BALANCED:
    k_n = np.arange(config["N"])+1
    i_vec = np.power(k_n, -config["kappa"])
    init_config["weight_vec"] = (i_vec/sum(i_vec)*100)

'''
Initialization
'''
if ALGORITHM == 'DIV_AOI' or ALGORITHM == 'DIV_AOI_SCA':
    gamma_matrix_inf, price_inf, random_action_inf = get_lagrangian_infinite_mix(config,init_config)

if INIT_METHOD == "RR":
    ITER_FP = np.ceil(config["N"]/config["M"]).astype(int)
else:
    ITER_FP = param["ITER_FP"]

list_dls_train, list_dls_test = get_dataloaders(DATASET, BATCH_SIZE)
model_0 = load_model(DATASET, RANDOM_SEED)
print(model_0)
model = model_0
training_sets = list_dls_train
testing_sets = list_dls_test
loss_f = loss_classifier
gradients = get_gradients("clustered_2", model, [model] * N_CLIENT)
gradient_dim = len(get_flat_gradient(gradients[0]))
full_gradient_matrix = np.zeros((N_CLIENT, gradient_dim))
n_samples = np.array([len(db.dataset) for db in training_sets])
weights = n_samples / np.sum(n_samples)
print("Clients' weights:", weights)
AoI_vector = np.ones((N_CLIENT, 1))
loss_hist = np.zeros((N_ITER + 1, N_CLIENT))
acc_hist = np.zeros((N_ITER + 1, N_CLIENT))
AoI_hist = np.zeros((N_ITER + 1, N_CLIENT))
trans_time_hist = np.zeros((N_ITER + 1, ))
update_vector = np.zeros((N_CLIENT,))

for k, dl in enumerate(training_sets):
    loss_hist[0, k] = float(loss_dataset(model, dl, loss_f).detach())
    acc_hist[0, k] = accuracy_dataset(model, dl)


# LOSS AND ACCURACY OF THE INITIAL MODEL
server_loss = np.dot(weights, loss_hist[0])
server_acc = np.dot(weights, acc_hist[0])
print(f"====> i: 0 Loss: {server_loss} Test Accuracy: {server_acc}")
sampled_clients_hist = np.zeros((N_ITER, N_CLIENT)).astype(int)
gradients = get_gradients("clustered_2", model, [model] * N_CLIENT)
full_gradient_matrix = np.zeros((N_CLIENT, gradient_dim))

'''
Save Configurations
'''
run_name = DATASET + '_' + ALGORITHM + '_' + INIT_METHOD  + f'_ran{RANDOM_SEED}'
if ALGORITHM == "DIV_AOI" or ALGORITHM == "DIV_AOI_SCA": 
    run_name += f'_factor{FACTOR}_lam{LAMBDAA}_AoI{config["USE_AOI"]}' 
saveDate = datetime.now()
fm_dt = saveDate.strftime("%Y%m%d")
save_path = 'results/'+fm_dt
os.makedirs(save_path, exist_ok=True)
file_name = save_path +"/"+run_name+'.mat'
print('Outputs are saved at ' + file_name)


if USE_WANDB:
    wandb.init(project=WB_PROJECT, name=run_name, entity="manyouma") #812f88afb619e5e755875c2e543c3b407a047b21
    wandb.config = {
    "algorithm": ALGORITHM,
    "dataset": DATASET,
    "config": config,
    }

# %%
if OPTIMIZER == 'GUROBI':
    import gurobipy
    gurobi_env = gurobipy.Env()
    gurobi_env.setParam('TimeLimit', param["GB_TIMEOUT"]) # in seconds

# %%
'''
FL Training Loop
'''
for i in range(N_ITER):
    channel_level = sample_c_vec(config).astype(int)
    previous_global_model = deepcopy(model)
    clients_params = []
    clients_models = []
    sampled_clients_for_grad = []

    if i < ITER_FP:
        print('Warming up with MD')
        if INIT_METHOD == "MD":
            sampled_clients = np.random.choice(N_CLIENT, size=M_SELECTED, replace=True, p=weights)
        else:
            sampled_clients = np.mod(np.arange(i*config["M"], (i+1)*config["M"]), config["N"])
    else:
        AoI_input = (AoI_vector.flatten()-1).astype(int)
        AoI_input[AoI_input>config["A_max"]-1] = config["A_max"]-1
        index_vec = np.zeros((config['N'],))
        
        if ALGORITHM == 'MD':
            sampled_clients = np.random.choice(N_CLIENT, size=M_SELECTED, replace=True, p=weights)
        elif ALGORITHM == 'CS':
            if i < 1:
                sampled_clients = np.random.choice(N_CLIENT, size=M_SELECTED, replace=True, p=weights)
            else:
                linkage_matrix = linkage(sim_matrix, "ward")
                distri_clusters = get_clusters_with_alg2(
                    linkage_matrix, M_SELECTED, weights
                )
                sampled_clients = sample_clients(distri_clusters)
        elif ALGORITHM == 'DIVFL':
            sampled_clients = per_slot_problem_Deploy_DivFL(config, full_gradient_matrix)
        elif ALGORITHM == 'DIV_AOI' or ALGORITHM == 'DIV_AOI_SCA':
            for i_user in np.arange(config["N"]):
                current_state = np.ravel_multi_index((AoI_input[i_user], channel_level[i_user]), (config["A_max"], config["c_bar"]))
                index_vec[i_user] = gamma_matrix_inf[i_user, current_state]          
            if ALGORITHM == 'DIV_AOI_SCA':
                sampled_clients = per_slot_problem_Deploy_Gurobi_linearSCA(config, index_vec, full_gradient_matrix, FACTOR, gurobi_env)
            else:
                sampled_clients = per_slot_problem_Deploy_Gurobi_linear(config, index_vec, full_gradient_matrix, FACTOR, gurobi_env)
        #(config, index_vec, full_gradient_matrix, 1)
    
    print(sampled_clients)
    
    AoI_vector += 1
    AoI_vector[sampled_clients] = 1
    update_vector[sampled_clients] += 1

    for k in sampled_clients:
        local_model = deepcopy(model)
        local_optimizer = optim.SGD(local_model.parameters(), lr=LEARNING_RATE)
        
        local_learning(
            local_model,
            MU,
            local_optimizer,
            training_sets[k],
            N_SGD,
            loss_f,
        )
        
        local_model_matrix = deepcopy(local_model)
        model_matrix = deepcopy(model)
        full_gradient_matrix[k,:] = get_flat_gradient_fromModel(local_model_matrix)-get_flat_gradient_fromModel(model_matrix)

        # SAVE THE LOCAL MODEL TRAINED
        list_params = list(local_model.parameters())
        list_params = [
            tens_param.detach() for tens_param in list_params
        ]
        clients_params.append(list_params)
        clients_models.append(deepcopy(local_model))

        sampled_clients_for_grad.append(k)
        sampled_clients_hist[i, k] = 1

    # CREATE THE NEW GLOBAL MODEL AND SAVE IT
    model = FedAvg_agregation_process(
        deepcopy(model), clients_params, weights=[1 / M_SELECTED] * M_SELECTED
    )

    for k, dl in enumerate(training_sets):
        loss_hist[i + 1, k] = float(
            loss_dataset(model, dl, loss_f).detach()
        )

    for k, dl in enumerate(testing_sets):
        acc_hist[i + 1, k] = accuracy_dataset(model, dl)
        AoI_hist[i + 1, k] = AoI_vector[k]
        trans_time_hist[i+1] += init_config["cost_matrix"][k, channel_level[k]]*sum(sampled_clients==k)

    server_loss = np.dot(weights, loss_hist[i + 1])
    server_acc = np.dot(weights, acc_hist[i + 1])

    print(f"====> i: {i+1} Loss: {server_loss} Server Test Accuracy: {server_acc}")
    if USE_WANDB:

        wandb.log({"server_loss": server_loss,
                   "server_acc": server_acc, 
                   "trans_time": trans_time_hist[i+1]})

    gradients_i = get_gradients(
        "clustered_2", previous_global_model, clients_models
    )
    for idx, gradient in zip(sampled_clients_for_grad, gradients_i):
        gradients[idx] = gradient
        full_gradient_matrix[idx,:] = get_flat_gradient(gradient)


    LEARNING_RATE *= DECAY

    if ALGORITHM == "CS":
        sim_matrix = get_matrix_similarity_from_grads(
            gradients, distance_type = "L2"
        )

    print(f'Difference: {nla.norm(np.mean(full_gradient_matrix, 0) - np.mean(full_gradient_matrix[sampled_clients,:],0))}')
    # ================== Save the Results ====================
    if i % 10 == 0:
        mdic = {"config": config,
            "init_config": init_config, 
            "loss_hist": loss_hist, 
            "acc_hist": acc_hist, 
            "trans_time_hist": trans_time_hist, 
            "i":i,
        }

        savemat(file_name, mdic)