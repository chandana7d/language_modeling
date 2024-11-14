import numpy as np
import itertools
np.random.seed(0)


class Parameters:
    
    @staticmethod
    def generate_true_dist_parameters(len_x, lb,ub):
        elements = [0, 1]
        dict_parameters = {"x1": {(): np.random.uniform(lb, ub)}}
        for k in range(2, len_x + 1):
            permutations = list(itertools.product(elements, repeat=k - 1))
            dict_parameters["x" + str(k)] = {perm: np.random.uniform(lb, ub) for perm in permutations}
        return dict_parameters
    
    @staticmethod
    def generate_logistic_parameters(len_x, lb,ub):
        elements = [0, 1]
        dict_parameters = {"x1": { "bias" : np.random.uniform(lb, ub)} }
        for k in range(2, len_x + 1):
            dict_parameters["x" + str(k)] = {"bias" :np.random.uniform(lb, ub)  }
            for j in range(1, k):
                dict_parameters["x" + str(k)]["x"+str(j)] = np.random.uniform(lb,ub)
        return dict_parameters


def sigmoid(x):
    return 1/(1+np.exp(-x))
    

def get_a_sample(parameters):
    x_sample = []
    p_chain  = []
    prob     = 1.0
    for i in range(len(parameters)):
        z = parameters["x"+str(i+1)]["bias"]
        for k in range(1,i+1):
            z += x_sample[k-1] * parameters["x"+str(i+1)]["x"+str(k)]
        p_est = sigmoid(z)
        s_i   = np.random.binomial(1,p_est)
        x_sample+=[s_i]
        prob *= p_est
        p_chain+=[float(p_est)]
    return x_sample, prob, p_chain


def get_n_samples(parameters,num_sample):
    samples = []
    for i in range(0,num_sample):
        x_sample, prob, _ = get_a_sample(parameters)
        samples += [x_sample]
    return samples


def get_prob(sub, parameters):
    
    pass
    


# log p = log(p(x1)) + log(p(x2|x1)) + ..... 

# ∇ log p(X) =  (1-σ1) * z1' + (1-σ2)* z2' + .......

# ∂log p(X)/∂b1 = (1-σ1) * 1  
# ∂log p(X)/∂b2 = (1-σ2) * 1  
# ∂log p(X)/∂a2 = (1-σ2) * x1 


def get_comp_prob(parameters, sub_sample = []):

    if len(sub_sample)==0:
        z = parameters["x1"]["bias"]
        p = sigmoid(z)
        return p 
    
    len_x=len(sub_sample)+1
    z = parameters["x"+str(len_x)]["bias"]
    for i in range(len(sub_sample)):
        z+= (sub_sample[i] * parameters["x"+str(len_x)]["x"+str(i+1)])
    return sigmoid(z)


def derivative_parameters(sample,parameters):
    dict_parameters = {}
    for k in range(len(sample)):
        p = get_comp_prob(parameters=parameters,sub_sample=sample[:k])
        dict_parameters["dl" + str(k+1)] = { "db" : (1-p)  }
        for j in range(k):
            dict_parameters["dl" + str(k+1)]["da"+str(j+1)] = sample[j] * (1-p)
    return dict_parameters


def initatize_derivative(len_sample):
    dict_parameters = {}
    for k in range(len_sample):
        dict_parameters["dl" + str(k+1)] = { "db" :0.0  }
        for j in range(k):
            dict_parameters["dl" + str(k+1)]["da"+str(j+1)] = 0.0
    return dict_parameters


def derivative_n_samples(samples, learning_params):
    len_sample = len(samples[0])
    n = len(samples)
    grad_dict = initatize_derivative(len_sample)

    for sample in samples:
        for k in range(len(sample)):
            #(sample)
            p = get_comp_prob(parameters=learning_params,sub_sample=sample[:k])
            grad_dict["dl" + str(k+1)]["db"] +=  (1-p)  
            for j in range(k):
                grad_dict["dl" + str(k+1)]["da"+str(j+1)] += sample[j] * (1-p)

    for k in range(len_sample):
        grad_dict["dl" + str(k+1)]["db"]= grad_dict["dl" + str(k+1)]["db"]/n   
        for j in range(k):
            grad_dict["dl" + str(k+1)]["da"+str(j+1)] = grad_dict["dl" + str(k+1)]["da"+str(j+1)]/n
    
    return grad_dict
    

def gradient_descent(samples, learning_params, lr):
    updated_parameters = {}
    dp = derivative_n_samples(samples,learning_params)
    len_sample  = len(samples[0])
    for k in range(len_sample):
        updated_parameters["x"+str(k+1)] = {"bias": learning_params["x"+str(k+1)]["bias"] + lr * dp["dl"+str(k+1)]["db"]}
        for j in range(1,k+1):
            updated_parameters["x"+str(k+1)]["x"+str(j)] = learning_params["x"+str(k+1)]["x"+str(j)] + lr * dp["dl"+str(k+1)]["da"+str(j)]
    return updated_parameters


parameters = Parameters().generate_logistic_parameters(len_x=5, lb=0.0, ub=2.0)
rand_parameters = Parameters().generate_logistic_parameters(len_x=5, lb=0.0, ub=2.2)

#Batch training 
sample_x = get_n_samples(parameters, 100000)

batch_size = 1000
num_batches = len(sample_x) // batch_size
num_epochs = 10

def print_parameters(p1, p2):
    for k in range(len(p1)):
            print(p1["x" + str(k+1)]["bias"] , p2["x" + str(k+1)]["bias"])
            for j in range(k):
                print(p1["x" + str(k+1)]["x"+str(j+1)], p2["x" + str(k+1)]["x"+str(j+1)] )
        
print("Before Training -----, ")
print_parameters(p1=parameters, p2=rand_parameters)



for epoch in range(1, num_epochs + 1):
    print(f"\nEpoch {epoch}/{num_epochs}")
    
    for i in range(num_batches):
        sample_data = sample_x[i * batch_size : batch_size * (i + 1)]
        # Calculate and update gradients
        deriv_sums =  derivative_n_samples(samples = sample_data, learning_params = rand_parameters )
        rand_parameters =  gradient_descent(samples = sample_data, learning_params = rand_parameters, lr= 0.01)

print("After Training -----, ")
print_parameters(p1=parameters, p2=rand_parameters)
