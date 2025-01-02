import numpy as np
import cvxpy as cp

def generate_frequency(config):
    kappa = config["kappa"]
    f = np.arange(config["N"])+1
    distribution = f.astype(float)**(-kappa)/sum(f.astype(float)**(-kappa))
    return distribution

def get_c_vec_stat(config):
    N = config["N"]
    c_bar = config["c_bar"]
    c_vec_stat = np.zeros((c_bar,))
    max_sample = 10000
    fading = np.random.rayleigh(1,(max_sample,))
    fading_max = np.percentile(np.random.rayleigh(1,(max_sample,)), 99)
    fading[fading>fading_max-0.00001] = fading_max -0.00001
    for i_c_bar in np.arange(c_bar):
        c_vec_stat[i_c_bar] = np.sum(np.floor(fading/fading_max*c_bar) == i_c_bar)
    c_vec_stat /= max_sample
    return c_vec_stat

def channel_model(config):
    np.random.seed(config["random_seed"])
    N = config["N"]
    c_bar = config["c_bar"]
    # Characteristics of a channel   
    Max_dist = 1500             #[km]
    noise_variance = -174       #[dBm]
    transmission_power = 28     #[dBm]
    BW = 50e6                   #[Hz]

    num_bits = 159800 
    distance_vector = 10+(Max_dist-10)*np.random.random((N,))
    pathloss = 128.1+37.6*np.log10(distance_vector/1000)
    channel_gain = np.sqrt(10**(-pathloss/10))
    noise_variance = 10**(noise_variance/10)/1000*BW
    P = 10**(transmission_power/10)/1000

    fading_max = np.percentile(np.random.rayleigh(1,(10000,)), 99)
    start = fading_max/(c_bar)/2
    c_vec = np.linspace(start,fading_max-start,c_bar)
    downlink_rate = np.log2(1+P*channel_gain**2/noise_variance)*BW
    finish_time = np.zeros((N, c_bar))
    for i_user in np.arange(N):
        for i_channel in np.arange(c_bar):
            current_rate = downlink_rate[i_user]*c_vec[i_channel]
            finish_time[i_user, i_channel] = num_bits/current_rate
    
    return finish_time


def sample_c_vec(config):
    N = config["N"]
    c_bar = config["c_bar"]
    fading = np.random.rayleigh(1,(N,))
    fading_max = np.percentile(np.random.rayleigh(1,(10000,)), 99)
    fading[fading>fading_max-0.00001] = fading_max -0.00001
    index = np.floor(fading/fading_max*c_bar)
    return index



def get_diversity_matrix(config):
    N_clients = config["N"]
    N_classes = 10
    np.random.seed(config["random_seed"])
    alpha = config["alpha"]
    sim_matrix = np.random.dirichlet([alpha]*N_classes, size = N_clients)#/np.sqrt(10)

    diversity_matrix = np.zeros((N_clients,N_clients))
    for n_1 in np.arange(N_clients):
        for n_2 in np.arange(N_clients):
            diversity_matrix[n_1, n_2] = np.sum((sim_matrix[n_1,:]-sim_matrix[n_2,:])**2)
    return diversity_matrix


def get_location_matrix(config):
    alpha =config["alpha"]
    N_clients = config["N"]
    N_classes = 10
    np.random.seed(config["random_seed"])
    matrix = np.random.dirichlet([alpha]*N_classes, size = N_clients)#/np.sqrt(10)
    return matrix


def get_c_vec_stat(config):
    N = config["N"]
    c_bar = config["c_bar"]
    c_vec_stat = np.zeros((c_bar,))
    max_sample = 1500
    fading = np.random.rayleigh(1,(max_sample,))
    fading_max = np.percentile(np.random.rayleigh(1,(max_sample,)), 99)
    fading[fading>fading_max-0.00001] = fading_max -0.00001
    for i_c_bar in np.arange(c_bar):
        c_vec_stat[i_c_bar] = np.sum(np.floor(fading/fading_max*c_bar) == i_c_bar)
    c_vec_stat /= max_sample
    return c_vec_stat
    

def generate_weight_vec(config):
    np.random.seed(config["random_seed"])
    weight_vec = np.random.random((config["N"],))*100
    return weight_vec