import numpy as np
import cvxpy as cp
from scipy.linalg import sqrtm
import numpy.linalg as nla
import gurobipy
from sklearn.metrics import pairwise_distances

def per_slot_problem_Deploy_DivFL(config, full_gradient_matrix):
    norm_diff = pairwise_distances(full_gradient_matrix)
    V_set = set(range(config["N"]))
    SUi = set()
    for ni in range(config["M"]):
        #print(f"ni:{ni}, V_set:{len(V_set)}, SUi:{(SUi)}")
        if config["N"] < len(V_set):
            R_set = np.random.choice(list(V_set), config["N"], replace=False)
        else:
            R_set = list(V_set)
        if ni == 0:
            marg_util = norm_diff[:, R_set].sum(0)
            i = marg_util.argmin()
            client_min = norm_diff[:, R_set[i]]
            #print(f"i:{i}, R_set:{(R_set)}, Included client: {R_set[i]}, client_min: {sla.norm(client_min)}")
        else:
            client_min_R = np.minimum(client_min[:,None], norm_diff[:,R_set])
            marg_util = client_min_R.sum(0)
            i = marg_util.argmin()
            client_min = client_min_R[:, i]
            #print(f"i:{i}, R_set:{(R_set)}, Included client: {R_set[i]}, client_min: {sla.norm(client_min)}")
        SUi.add(R_set[i])
        V_set.remove(R_set[i])
    return np.array(list(SUi))


def per_slot_problem_Deploy_Gurobi_linearSCA(config, index_vec, location_matrix, factor, gurobi_env):
    init_x = np.random.random((config["N"],))
    sample = np.argsort(np.random.random(config["N"]))[:config["M"]]
    init_x[sample] = 1
    penalty = 1
    for i in np.arange(10):
        coeff_1 = config["N"]-config["M"]
        coeff_2 = config["M"]
        x = cp.Variable(config["N"], integer=False)
        Gram = location_matrix@location_matrix.T
        half_Gram = sqrtm(Gram) 
        objective = cp.Minimize(-index_vec@x+factor*cp.norm(half_Gram@(coeff_1*x-coeff_2*(1-x)))-penalty*init_x.T@init_x.T-penalty*2*init_x.T@(x-init_x))
        constraints = [cp.sum(x)==config["M"], x<=1, x>=0] 
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.GUROBI, verbose=False)
        a = nla.norm((coeff_1*x.value.T-coeff_2*(1-x.value.T))@half_Gram)
        init_x = x.value
        penalty = penalty * 2

    #print(f'Min: {a}')
    return np.argsort(-x.value)[:10]



def per_slot_problem_Deploy_Gurobi_linear(config, index_vec, location_matrix, factor, gurobi_env):
    coeff_1 = config["N"]-config["M"]
    coeff_2 = config["M"]
    index = np.arange(config["N"])
    x = cp.Variable(config["N"], integer=True)
    Gram = location_matrix@location_matrix.T
    half_Gram = sqrtm(Gram) 
    objective = cp.Minimize(-index_vec@x+factor*(cp.norm(half_Gram@(coeff_1*x-coeff_2*(1-x)))))
    constraints = [cp.sum(x)==config["M"], x<=1, x>=0] 
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.GUROBI, env=gurobi_env)
    sol = x.value
    a = nla.norm((coeff_1*x.value.T-coeff_2*(1-x.value.T))@half_Gram)
    print(f'Min: {a}')

    return index[sol>0.9]


def per_slot_problem_Deploy(config, index_vec, location_matrix, factor):
    current_val = -100000
    for C in np.arange(config["M"]+1):
        x = cp.Variable(config["N"], integer=True)
        Gram = location_matrix@location_matrix.T
        half_Gram = sqrtm(Gram)
        objective = cp.Maximize(index_vec@x + factor*(2*config["eta"]*np.diag(Gram).transpose()@x*C-2*config["eta"]*cp.power(cp.norm(half_Gram@x),2)))
        constraints = [cp.sum(x) == C, x<=1, x>=0] 
        prob = cp.Problem(objective,constraints)
        prob.solve()
        if prob.value > current_val:
            current_val = prob.value
            sol = x.value
        index = np.arange(config["N"])
    return index[sol>0.9]




def per_slot_problem_DA(config, index_vec):
    current_val = -100000
    for C in np.arange(config["M"]+1):
        #print(f"Currrent_C: {C}")
        x = cp.Variable(config["N"], integer=True)
        #np.random.seed(config["random_seed"])
        #location_matrix = init_config["location_matrix"][i_time]#np.random.random((config["N"],2))
        objective = cp.Maximize(index_vec@x)
        constraints = [cp.sum(x) == C, x<=1, x>=0] 
        prob = cp.Problem(objective,constraints)
        prob.solve()
        #print(f"Solution: {x.value.astype(int)}, Value: {prob.value}")
        if prob.value > current_val:
            current_val = prob.value
            sol = x.value
        
        index = np.arange(config["N"])

    return index[sol>0.9]



def per_slot_problem(config, init_config, index_vec, factor):
    current_val = -100000
    for C in np.arange(config["M"]+1):
        #print(f"Currrent_C: {C}")
        x = cp.Variable(config["N"], integer=True)
        #np.random.seed(config["random_seed"])
        location_matrix = init_config["location_matrix"] #np.random.random((config["N"],2))
        Gram = location_matrix@location_matrix.T
        objective = cp.Maximize(index_vec@x + factor*(2*config["eta"]*np.diag(Gram).transpose()@x*C-2*config["eta"]*cp.power(cp.norm(location_matrix.transpose()@x),2))-0)
        constraints = [cp.sum(x) == C, x<=1, x>=0] 
        prob = cp.Problem(objective,constraints)
        prob.solve()
        #print(f"Solution: {x.value.astype(int)}, Value: {prob.value}")
        if prob.value > current_val:
            current_val = prob.value
            index = np.zeros(config["N"])
            index[x.value>0.95] = 1
        
    return index



def get_lagrangian_finite(config, init_config):
    N = config["N"]
    M = config["M"]
    T = config["T"]
    weight_vec = init_config["weight_vec"]
    c_bar = config["c_bar"]
    finish_time = init_config["cost_matrix"][:,:]
    A_max = config["A_max"]
    num_action = config["num_action"]
    num_state = A_max*c_bar
    
    '''
    Construct the reward vectors and the state transition matrices
    '''
    Reward_vector = np.zeros((N, A_max, c_bar, num_action))
    Reward_vector_A = np.zeros((N, A_max, c_bar, num_action))
    P_matrix = np.zeros((A_max, c_bar, A_max, c_bar, num_action))
    for i_user in np.arange(N):
        for i_age in np.arange(A_max):
            for i_c in np.arange(c_bar):  
                for i_action in np.arange(num_action):
                    Reward_vector[i_user, i_age, i_c, i_action] = - config["lambda"]*finish_time[i_user, i_c]*i_action
                    Reward_vector_A[i_user, i_age, i_c, i_action] = -(i_age + 1) * weight_vec[i_user]
                    #if i_age == A_max-1:
                    #    Reward_vector_A[i_user, i_age, i_c, i_action] = -1000000
                    for i_age2 in np.arange(A_max):
                        for i_c2 in np.arange(c_bar):
                            if i_action == 0:
                                if i_age2 == min(A_max-1, i_age + 1):
                                    P_matrix[i_age, i_c, i_age2, i_c2, i_action] = init_config["c_vec_stat"][i_c2]
                            if i_action == 1:
                                if i_age2 == 0:
                                    P_matrix[i_age, i_c, i_age2, i_c2, i_action] = init_config["c_vec_stat"][i_c2]

    Reward_vector_F =  np.zeros((N, num_state, num_action))
    Reward_vector_A_F =  np.zeros((N, num_state, num_action))
    P_matrix_F = np.zeros((num_state, num_state, num_action))    

    for i_user in np.arange(N):
        for i_age in np.arange(A_max):
            for i_c in np.arange(c_bar):
                state_1 = np.ravel_multi_index((i_age, i_c), (A_max, c_bar))
                for i_action in np.arange(num_action):
                    Reward_vector_F[i_user, state_1, i_action] = Reward_vector[i_user, i_age, i_c, i_action]
                    Reward_vector_A_F[i_user, state_1, i_action] = Reward_vector_A[i_user, i_age, i_c, i_action]
                    for i_age2 in np.arange(A_max):
                        for i_c2 in np.arange(c_bar):
                            state_2 = np.ravel_multi_index((i_age2, i_c2), (A_max, c_bar))
                            P_matrix_F[state_1, state_2, i_action] = P_matrix[i_age, i_c, i_age2, i_c2, i_action]
    
    '''
    solve the linear program
    '''
    # Start to solve the CVX problem
    x_0 = cp.Variable(T*N*num_state)
    x_1 = cp.Variable(T*N*num_state) 
    obj = 0
    Constraint = []
    i_constraint = 0
    Lagrange_loc = np.zeros((T, ))

    for i_T in np.arange(T):
        total_freq = 0
        for i_user in np.arange(N):
            obj_index = np.zeros((T, N, num_state))
            obj_index_prev = np.zeros((T, N, num_state))
            for i_state in np.arange(num_state):
                obj_index[i_T, i_user, i_state] = 1
                if i_T > 0:
                    obj_index_prev[i_T-1, i_user, i_state] = 1
            obj_index = (obj_index == 1) 
            obj_index_prev = (obj_index_prev == 1)
            obj += Reward_vector_F[i_user, :,0].flatten().T@x_0[obj_index.flatten()].flatten() \
                      + Reward_vector_F[i_user, :,1].flatten().T@x_1[obj_index.flatten()].flatten() \
                      + Reward_vector_A_F[i_user, :,0].flatten().T@x_0[obj_index.flatten()].flatten() \
                      + Reward_vector_A_F[i_user, :,1].flatten().T@x_1[obj_index.flatten()].flatten() 

            if i_T > 0:
                # ------- Constraint C2 -----------------
                Constraint += [x_0[obj_index.flatten()].flatten() + x_1[obj_index.flatten()].flatten() ==
                                        + (P_matrix_F[:,:,0].squeeze().T)@x_0[obj_index_prev.flatten()].flatten() 
                                        + (P_matrix_F[:,:,1].squeeze().T)@x_1[obj_index_prev.flatten()].flatten() 
                ]
                
                i_constraint += 1
                
            total_freq += np.ones((num_state,))@x_1[obj_index.flatten()]
        # ------------- Constraint C1b  -----------------  
        Constraint += [total_freq <= M]
        Lagrange_loc[i_T] = i_constraint
        i_constraint += 1


    for i_user in np.arange(N):
        cons_index = np.zeros((T, N, num_state))
        #for i_state in np.arange(num_state):
        cons_index[0, i_user, 0] = 1
        cons_index = (cons_index == 1)
        # -------------- Constraint C4 ------------------
        Constraint += [x_0[cons_index.flatten()]+x_1[cons_index.flatten()] == 1]            
        i_constraint += 1
        
        
    Objective = cp.Maximize(obj)
    # --------------- Constraint C5 --------------------
    Constraint += [x_0 >= 0]
    i_constraint += 1
    Constraint += [x_1 >= 0]
    i_constraint += 1
    prob = cp.Problem(Objective, Constraint)
    prob.solve(solver=cp.GUROBI)
    
    
    '''
    Recover the Lagrange multipliers at different times
    '''
    price_vec = np.zeros((T, ))
    for i_T, i_loc in enumerate(Lagrange_loc):
        price_vec[i_T] = prob.constraints[i_loc.astype(int)].dual_value
    price_vec = np.maximum(price_vec, np.zeros_like(price_vec))
    
    '''
    Get the Optimal Lagrangian Index
    '''
    gamma_matrix = np.zeros((config["T"], config["N"], num_state))
    for i_user in np.arange(config["N"]):
        Value_Function = np.zeros((num_state,))
        for i_T in np.arange(T):
            current_T = T-i_T-1

            Value_Function_0 =  Reward_vector_F[i_user, :, 0] + Reward_vector_A_F[i_user,:,0] + P_matrix_F[:,:,0].squeeze()@Value_Function
            Value_Function_1 =  -price_vec[current_T] + Reward_vector_F[i_user, :, 1] + Reward_vector_A_F[i_user,:,1] + P_matrix_F[:,:,1].squeeze()@Value_Function 
            Value_Function_new = np.maximum(Value_Function_0, Value_Function_1)

            dual_mu = Value_Function 
            gamma_0 =  P_matrix_F[:,:,0].squeeze()@dual_mu + Reward_vector_F[i_user,:,0]
            gamma_1 =  P_matrix_F[:,:,1].squeeze()@dual_mu + Reward_vector_F[i_user,:,1] #- price_vec[current_T])
            gamma_ind = gamma_1.flatten() - gamma_0.flatten()
            gamma_matrix[current_T, i_user, :] = gamma_ind.flatten()

            Value_Function = Value_Function_new
            
    update_0 = x_0.value
    update_1 = x_1.value

    random_action = 0.5*np.ones((T, N, num_state))
    for i_t in np.arange(config["T"]):
        for i_n in np.arange(config["N"]):
            for i_state in np.arange(num_state):
                state_loc = np.ravel_multi_index((i_t, i_n, i_state), (T, N, num_state))
                if update_0[state_loc] + update_1[state_loc] > 0.000001:
                    random_action[i_t, i_n, i_state] = update_1[state_loc]/(update_0[state_loc] + update_1[state_loc])

    return gamma_matrix, price_vec, random_action


def get_lagrangian_infinite(config, init_config):
    N = config["N"]
    M = config["M"]
    T = config["T"]
    weight_vec = init_config["weight_vec"]
    c_bar = config["c_bar"]
    finish_time = init_config["cost_matrix"][:,:]
    A_max = config["A_max"]
    num_action = config["num_action"]
    num_state = A_max*c_bar

    '''
    Construct the reward vectors and the state transition matrices
    '''
    Reward_vector = np.zeros((N, A_max, c_bar, num_action))
    Reward_vector_A = np.zeros((N, A_max, c_bar, num_action))
    P_matrix = np.zeros((A_max, c_bar, A_max, c_bar, num_action))
    for i_user in np.arange(N):
        for i_age in np.arange(A_max):
            for i_c in np.arange(c_bar):  
                for i_action in np.arange(num_action):
                    Reward_vector[i_user, i_age, i_c, i_action] =  (- config["lambda"]*finish_time[i_user, i_c])*i_action
                    Reward_vector_A[i_user, i_age, i_c, i_action] = -(i_age + 1) * weight_vec[i_user]#*(1-i_action)
                   # if i_age == A_max-1:
                     #   Reward_vector_A[i_user, i_age, i_c, i_action] = -1000000
                    for i_age2 in np.arange(A_max):
                        for i_c2 in np.arange(c_bar):
                            if i_action == 0:
                                if i_age2 == min(A_max-1, i_age + 1):
                                    P_matrix[i_age, i_c, i_age2, i_c2, i_action] = init_config["c_vec_stat"][i_c2]
                            if i_action == 1:
                                if i_age2 == 0:
                                    P_matrix[i_age, i_c, i_age2, i_c2, i_action] = init_config["c_vec_stat"][i_c2]

    Reward_vector_F =  np.zeros((N, num_state, num_action))
    Reward_vector_A_F =  np.zeros((N, num_state, num_action))
    P_matrix_F = np.zeros((num_state, num_state, num_action))    

    for i_user in np.arange(N):
        for i_age in np.arange(A_max):
            for i_c in np.arange(c_bar):
                state_1 = np.ravel_multi_index((i_age, i_c), (A_max, c_bar))
                for i_action in np.arange(num_action):
                    Reward_vector_F[i_user, state_1, i_action] = Reward_vector[i_user, i_age, i_c, i_action]
                    Reward_vector_A_F[i_user, state_1, i_action] = Reward_vector_A[i_user, i_age, i_c, i_action]
                    for i_age2 in np.arange(A_max):
                        for i_c2 in np.arange(c_bar):
                            state_2 = np.ravel_multi_index((i_age2, i_c2), (A_max, c_bar))
                            P_matrix_F[state_1, state_2, i_action] = P_matrix[i_age, i_c, i_age2, i_c2, i_action]

    '''
    solve the linear program
    '''
    # Start to solve the CVX problem
    x_0 = cp.Variable(N*num_state)
    x_1 = cp.Variable(N*num_state) 
    Constraint = []
    i_constraint = 0



    total_freq = 0
    obj=0

    for i_user in np.arange(N):

        obj_index = np.zeros((N, num_state))
        for i_state in np.arange(num_state):
            obj_index[i_user, i_state] = 1

        obj_index = (obj_index == 1) 

        obj += Reward_vector_F[i_user, :,0].flatten().T@x_0[obj_index.flatten() ] \
                  + Reward_vector_F[i_user, :,1].flatten().T@x_1[obj_index.flatten() ] \
                  + Reward_vector_A_F[i_user, :,0].flatten().T@x_0[obj_index.flatten() ] \
                  + Reward_vector_A_F[i_user, :,1].flatten().T@x_1[obj_index.flatten() ] 



        Constraint += [x_0[obj_index.flatten()]+ x_1[obj_index.flatten()] ==
                                + (P_matrix_F[:,:,0].squeeze().T)@x_0[obj_index.flatten() ].flatten() 
                                + (P_matrix_F[:,:,1].squeeze().T)@x_1[obj_index.flatten() ].flatten() 
        ]


        total_freq += np.ones((num_state,))@x_1[obj_index.flatten()]
        Constraint += [np.ones((num_state,))@(x_1[obj_index.flatten()]+x_0[obj_index.flatten()]) == 1]     

    Constraint += [total_freq <= M]



    Objective = cp.Maximize(obj)
    Constraint += [x_0 >= 0]
    Constraint += [x_1 >= 0]

    prob = cp.Problem(Objective, Constraint)
    prob.solve(solver=cp.GUROBI)
    
    lagrange_price = prob.constraints[-3].dual_value
    
    update_0 = x_0.value
    update_1 = x_1.value

    random_action = 0.5*np.ones((N, num_state))
    for i_t in np.arange(config["T"]):
        for i_n in np.arange(config["N"]):
            for i_state in np.arange(num_state):
                state_loc = np.ravel_multi_index((i_n, i_state), (N, num_state))
                if update_0[state_loc] + update_1[state_loc] > 0.000001:
                    random_action[i_n, i_state] = update_1[state_loc]/(update_0[state_loc] + update_1[state_loc])


    gamma_matrix = np.zeros((config["N"], num_state))

    for i_user in np.arange(config["N"]):
        Value_Function = np.zeros((num_state,))
        for i_T in np.arange(1000):
            Value_Function_0 =  Reward_vector_F[i_user, :, 0] + Reward_vector_A_F[i_user,:,0] + P_matrix_F[:,:,0].squeeze()@Value_Function
            Value_Function_1 =  -lagrange_price + Reward_vector_F[i_user, :, 1] + Reward_vector_A_F[i_user,:,1] + P_matrix_F[:,:,1].squeeze()@Value_Function 
            Value_Function = np.maximum(Value_Function_0, Value_Function_1)
            Value_Function -= Value_Function[0]

        dual_mu = Value_Function 
        gamma_0 =  P_matrix_F[:,:,0].squeeze()@dual_mu + Reward_vector_F[i_user,:,0] + Reward_vector_A_F[i_user,:,0]
        gamma_1 =   P_matrix_F[:,:,1].squeeze()@dual_mu + Reward_vector_F[i_user,:,1] + Reward_vector_A_F[i_user,:,1] #- price_vec[current_T])
        gamma_ind = gamma_1.flatten() - gamma_0.flatten()
        gamma_matrix[i_user, :] = gamma_ind.flatten()
        
    return gamma_matrix, lagrange_price, random_action



def get_lagrangian_infinite_mix(config, init_config):
    N = config["N"]
    M = config["M"]
    T = config["T"]
    weight_vec = init_config["weight_vec"]
    c_bar = config["c_bar"]
    finish_time = init_config["cost_matrix"][:,:]
    A_max = config["A_max"]
    num_action = config["num_action"]
    num_state = A_max*c_bar

    '''
    Construct the reward vectors and the state transition matrices
    '''
    Reward_vector = np.zeros((N, A_max, c_bar, num_action))
    Reward_vector_A = np.zeros((N, A_max, c_bar, num_action))
    P_matrix = np.zeros((A_max, c_bar, A_max, c_bar, num_action))
    for i_user in np.arange(N):
        for i_age in np.arange(A_max):
            for i_c in np.arange(c_bar):  
                for i_action in np.arange(num_action):
                    Reward_vector[i_user, i_age, i_c, i_action] =  (- config["lambda"]*finish_time[i_user, i_c])*i_action
                    Reward_vector_A[i_user, i_age, i_c, i_action] = -(i_age + 1) * weight_vec[i_user]*config["USE_AOI"]
                   # if i_age == A_max-1:
                     #   Reward_vector_A[i_user, i_age, i_c, i_action] = -1000000
                    for i_age2 in np.arange(A_max):
                        for i_c2 in np.arange(c_bar):
                            if i_action == 0:
                                if i_age2 == min(A_max-1, i_age + 1):
                                    P_matrix[i_age, i_c, i_age2, i_c2, i_action] = init_config["c_vec_stat"][i_c2]
                            if i_action == 1:
                                if i_age2 == 0:
                                    P_matrix[i_age, i_c, i_age2, i_c2, i_action] = init_config["c_vec_stat"][i_c2]

    Reward_vector_F =  np.zeros((N, num_state, num_action))
    Reward_vector_A_F =  np.zeros((N, num_state, num_action))
    P_matrix_F = np.zeros((num_state, num_state, num_action))    

    for i_user in np.arange(N):
        for i_age in np.arange(A_max):
            for i_c in np.arange(c_bar):
                state_1 = np.ravel_multi_index((i_age, i_c), (A_max, c_bar))
                for i_action in np.arange(num_action):
                    Reward_vector_F[i_user, state_1, i_action] = Reward_vector[i_user, i_age, i_c, i_action]
                    Reward_vector_A_F[i_user, state_1, i_action] = Reward_vector_A[i_user, i_age, i_c, i_action]
                    for i_age2 in np.arange(A_max):
                        for i_c2 in np.arange(c_bar):
                            state_2 = np.ravel_multi_index((i_age2, i_c2), (A_max, c_bar))
                            P_matrix_F[state_1, state_2, i_action] = P_matrix[i_age, i_c, i_age2, i_c2, i_action]

    '''
    solve the linear program
    '''
    # Start to solve the CVX problem
    x_0 = cp.Variable(N*num_state)
    x_1 = cp.Variable(N*num_state) 
    Constraint = []
    i_constraint = 0
    total_freq = 0
    obj=0

    for i_user in np.arange(N):
        obj_index = np.zeros((N, num_state))
        for i_state in np.arange(num_state):
            obj_index[i_user, i_state] = 1

        obj_index = (obj_index == 1) 

        obj += Reward_vector_F[i_user, :,0].flatten().T@x_0[obj_index.flatten() ] \
                  + Reward_vector_F[i_user, :,1].flatten().T@x_1[obj_index.flatten() ] \
                  + Reward_vector_A_F[i_user, :,0].flatten().T@x_0[obj_index.flatten() ] \
                  + Reward_vector_A_F[i_user, :,1].flatten().T@x_1[obj_index.flatten() ] 

        Constraint += [x_0[obj_index.flatten()]+ x_1[obj_index.flatten()] ==
                                + (P_matrix_F[:,:,0].squeeze().T)@x_0[obj_index.flatten() ].flatten() 
                                + (P_matrix_F[:,:,1].squeeze().T)@x_1[obj_index.flatten() ].flatten() 
        ]

        total_freq += np.ones((num_state,))@x_1[obj_index.flatten()]
        Constraint += [np.ones((num_state,))@(x_1[obj_index.flatten()]+x_0[obj_index.flatten()]) == 1]     

    Constraint += [total_freq <= M]
    Objective = cp.Maximize(obj)
    Constraint += [x_0 >= 0]
    Constraint += [x_1 >= 0]

    prob = cp.Problem(Objective, Constraint)
    prob.solve(solver=cp.GUROBI)
    lagrange_price = prob.constraints[-3].dual_value
    update_0 = x_0.value
    update_1 = x_1.value
    random_action = 0.5*np.ones((N, num_state))
    for i_t in np.arange(config["T"]):
        for i_n in np.arange(config["N"]):
            for i_state in np.arange(num_state):
                state_loc = np.ravel_multi_index((i_n, i_state), (N, num_state))
                if update_0[state_loc] + update_1[state_loc] > 0.000001:
                    random_action[i_n, i_state] = update_1[state_loc]/(update_0[state_loc] + update_1[state_loc])

    gamma_matrix = np.zeros((config["N"], num_state))

    for i_user in np.arange(config["N"]):
        Value_Function = np.zeros((num_state,))
        for i_T in np.arange(1000):
            Value_Function_0 =  Reward_vector_F[i_user, :, 0] + Reward_vector_A_F[i_user,:,0] + P_matrix_F[:,:,0].squeeze()@Value_Function
            Value_Function_1 =  -lagrange_price + Reward_vector_F[i_user, :, 1] + Reward_vector_A_F[i_user,:,1] + P_matrix_F[:,:,1].squeeze()@Value_Function 
            Value_Function = np.maximum(Value_Function_0, Value_Function_1)
            Value_Function -= Value_Function[0]

        dual_mu = Value_Function 
        gamma_0 =  P_matrix_F[:,:,0].squeeze()@dual_mu + Reward_vector_F[i_user,:,0] + Reward_vector_A_F[i_user,:,0]
        gamma_1 =   P_matrix_F[:,:,1].squeeze()@dual_mu + Reward_vector_F[i_user,:,1] + Reward_vector_A_F[i_user,:,1] #- price_vec[current_T])
        gamma_ind = gamma_1.flatten() - gamma_0.flatten()
        gamma_matrix[i_user, :] = gamma_ind.flatten()
        
    return gamma_matrix, lagrange_price, random_action