from distutils.file_util import copy_file
import cvxpy as cp
import numpy as np
from scipy.io import savemat, loadmat
from util.util_AoI import channel_model, get_c_vec_stat, get_diversity_matrix, sample_c_vec 
from util.util_lagrangian import get_lagrangian_finite, get_lagrangian_infinite, per_slot_problem_DA
from scipy.io import savemat
import sys

T = 30
config={
    "eta": 0,                               # Panelty factor used for evaluation
    "N": int(sys.argv[1])*5,                # Number of total users
    "A_max": 20,                            # Max AoI considered
    "c_bar": 5,                             # Discretized version of channels
    "num_action": 2,                        # Number of actons per user
    "M": int(sys.argv[1]),                  # Number of sampled users
    "random_seed": 0,
    "T": T, 
    "kappa": 0.,
    "lambda": 200.,                         # Factor in front of the cost
    "alpha": 0.1,
}
NUM_SIM = 1000
      
print(config["M"])
print(config["N"])

init_config = {
    "cost_matrix": channel_model(config),
    "c_vec_stat": get_c_vec_stat(config),
    "weight_vec": np.ones((config["N"],)), #generate_weight_vec(config),
    "diversity_matrix": get_diversity_matrix(config)
} 

time_index = np.flip(np.arange(T),0)
next_bar_index = (config["c_bar"],)*config["N"]
N_next_cbar = config["c_bar"]**config["N"]


gamma_matrix, price_vec, random_action = get_lagrangian_finite(config,init_config)
gamma_matrix_inf, price_inf, random_action_inf = get_lagrangian_infinite(config,init_config)

print('start random')

'''
Test Random Scheduling
'''
# First dimension is age 
# Second dimension is diversity 
# Third dimension is cost
results_random = np.zeros((NUM_SIM, 3))
for i_seed in np.arange(NUM_SIM):
    np.random.seed(i_seed)
    random_vec = (np.random.random((config['T'], ))*config['N']).astype(int)

    current_age = np.zeros((config['N'], ), dtype=int)
    current_channel = np.zeros((config['N'],),dtype=int)
    for i_time in np.arange(config["T"]):
        index_vec = np.zeros((config['N'],))
        for i_user in np.arange(config["N"]):
            current_state = np.ravel_multi_index((current_age[i_user], current_channel[i_user]), (config["A_max"], config["c_bar"]))
            index_vec[i_user] = random_action[i_time, i_user, current_state]
            results_random[i_seed, 0] -= current_age[i_user]+1
            decision = np.random.random((config["N"],)) <= index_vec
        decision_index = np.arange(config["N"])[decision==1] 
        decision_matrix = np.expand_dims(decision,0)
        results_random[i_seed, 1] += config["eta"]*decision_matrix @ init_config["diversity_matrix"] @ decision_matrix.T
        decision = decision_index
        #print(f"time: {i_time} current age: {current_age}, channel: {current_channel}, actions: {decision+1}")
        for i_user in np.arange(config["N"]):

            if np.isin(i_user, np.array(decision)):
                results_random[i_seed, 2] -= config["lambda"]*init_config["cost_matrix"][i_user][current_channel[i_user]]
                current_age[i_user] = 0
            else:
                current_age[i_user] = min(current_age[i_user]+1, config["A_max"]-1)
        np.random.seed(random_vec[i_time])        
        current_channel = sample_c_vec(config).astype(int)

print('start finite')
'''
test the finite-horizon approach
'''
# First dimension is age 
# Second dimension is diversity 
# Third dimension is cost
results_finite = np.zeros((NUM_SIM, 3))
for i_seed in np.arange(NUM_SIM):
    np.random.seed(i_seed)
    random_vec = (np.random.random((config['T'], ))*config['N']).astype(int)
    current_age = np.zeros((config['N'], ), dtype=int)
    current_channel = np.zeros((config['N'],),dtype=int)
    for i_time in np.arange(config["T"]):
        index_vec = np.zeros((config['N'],))
        for i_user in np.arange(config["N"]):
            current_state = np.ravel_multi_index((current_age[i_user], current_channel[i_user]), (config["A_max"], config["c_bar"]))
            index_vec[i_user] = gamma_matrix[i_time, i_user, current_state]
            results_finite[i_seed, 0] -= current_age[i_user]+1
            
        decision = per_slot_problem_DA(config, index_vec)

        for i_user in np.arange(config["N"]):

            if np.isin(i_user, np.array(decision)):
                #reward -= config["lambda"]*init_config["cost_matrix"][i_user][current_channel[i_user]]
                current_age[i_user] = 0
                results_finite[i_seed, 2] -= config["lambda"]*init_config["cost_matrix"][i_user][current_channel[i_user]]
            else:
                current_age[i_user] = min(current_age[i_user]+1, config["A_max"]-1)
        np.random.seed(random_vec[i_time])        
        current_channel = sample_c_vec(config).astype(int)



print('start infinite')
'''
test the infite-horizon approach
'''
# First dimension is age 
# Second dimension is diversity 
# Third dimension is cost
results_infinite = np.zeros((NUM_SIM, 3))
for i_seed in np.arange(NUM_SIM):
    np.random.seed(i_seed)
    random_vec = (np.random.random((config['T'], ))*config['N']).astype(int)
    current_age = np.zeros((config['N'], ), dtype=int)
    current_channel = np.zeros((config['N'],),dtype=int)
    for i_time in np.arange(config["T"]):
        index_vec = np.zeros((config['N'],))
        for i_user in np.arange(config["N"]):
            current_state = np.ravel_multi_index((current_age[i_user], current_channel[i_user]), (config["A_max"], config["c_bar"]))
            index_vec[i_user] = gamma_matrix_inf[i_user, current_state]
            
            results_infinite[i_seed, 0] -= (current_age[i_user]+1)
            
        decision = per_slot_problem_DA(config, index_vec)
        #decision_index = np.arange(config["N"])[decision==1] 
        #decision_matrix = np.expand_dims(decision,0)
        #results_infinite[i_seed, 1] += config["eta"]*decision_matrix @ init_config["diversity_matrix"] @ decision_matrix.T
        #decision = decision_index
        #print(f"time: {i_time} current age: {current_age}, channel: {current_channel}, actions: {decision+1}")
        for i_user in np.arange(config["N"]):

            if np.isin(i_user, np.array(decision)):
                #reward -= config["lambda"]*init_config["cost_matrix"][i_user][current_channel[i_user]]
                current_age[i_user] = 0
                results_infinite[i_seed, 2] -= config["lambda"]*init_config["cost_matrix"][i_user][current_channel[i_user]]
            else:
                current_age[i_user] = min(current_age[i_user]+1, config["A_max"]-1)
        np.random.seed(random_vec[i_time])        
        current_channel = sample_c_vec(config).astype(int)

    #print(f"Reward: {reward}, Age: {reward_A}, diversity: {reward_D}, cost: {reward_C}")



mdic = {"config": config,
        "init_config": init_config, 
        "results_random": results_random, 
        "price_vec": price_vec,
        "price_inf": price_inf,
        "results_finite": results_finite, 
        "results_infinite": results_infinite, 
        "NUM_SIM": NUM_SIM,
       }
j = config["lambda"]
filename = f"results/Fig5_InfiniteHorizon_N{config['N']}_M{config['M']}_eta{config['eta']}_cbar{config['c_bar']}_lambda{j}.mat"
savemat(filename, mdic)






