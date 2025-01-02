import json

param = {
    "ALGORITHM": 'DIVFL', # DIV_AOI, CS, 
    "WB_PROJECT": 'MAR_28',
    "GB_TIMEOUT": 30,  
    "USE_WANDB": False,                    
    "OPTIMIZER": 'GUROBI',
    "DATASET": 'CIFAR10_bbal_100',
    'AOI': True, 
    'INIT_METHOD': 'RR',
    'FACTOR': 5,
    'LAMBDAA': 100000,
    'ETA': 20,
    'N_CLIENT': 100,
    'M_SELECTED': 10,
    'RANDOM_SEED': 100,
    'C_BAR': 20,
    'A_MAX': 40,
    'KAPPA': 0,
    'N_SGD': 50,
    'LEARNING_RATE': 0.01,
    'DECAY': 1.0,
    'MU': 0.0,
    "BATCH_SIZE": 50,
    "MNIST_ITER": 1010,
    "CIFAR10_ITER": 4010,
    "ITER_FP":1,
}


with open("config_beluga/CIFAR10_DIVFL_IID.json", "w") as write_file:
    json.dump(param, write_file, indent=4)