import os
import sys
import numpy as np
from copy import deepcopy
import random
from scipy.cluster.hierarchy import linkage

import torch
import torch.nn as nn
import torch.optim as optim

def set_to_zero_model_weights(model):
    """Set all the parameters of a model to 0"""
    for layer_weigths in model.parameters():
        layer_weigths.data.sub_(layer_weigths.data)



def FedAvg_agregation_process(model, clients_models_hist: list, weights: list):
    """Creates the new model of a given iteration with the models of the other
    clients"""

    new_model = deepcopy(model)
    set_to_zero_model_weights(new_model)

    for k, client_hist in enumerate(clients_models_hist):

        for idx, layer_weights in enumerate(new_model.parameters()):

            contribution = client_hist[idx].data * weights[k]
            layer_weights.data.add_(contribution)

    return new_model



def FedAvg_agregation_process_for_FA_sampling(
    model, clients_models_hist: list, weights: list
):
    """Creates the new model of a given iteration with the models of the other
    clients"""

    new_model = deepcopy(model)

    for layer_weigths in new_model.parameters():
        layer_weigths.data.sub_(sum(weights) * layer_weigths.data)

    for k, client_hist in enumerate(clients_models_hist):

        for idx, layer_weights in enumerate(new_model.parameters()):

            contribution = client_hist[idx].data * weights[k]
            layer_weights.data.add_(contribution)

    return new_model



def accuracy_dataset(model, dataset):
    """Compute the accuracy of `model` on `test_data`"""

    correct = 0

    for features, labels in dataset:

        predictions = model(features)
        _, predicted = predictions.max(1, keepdim=True)

        correct += torch.sum(predicted.view(-1, 1) == labels.view(-1, 1)).item()

    accuracy = 100 * correct / len(dataset.dataset)

    return accuracy


def loss_dataset(model, train_data, loss_f):
    """Compute the loss of `model` on `test_data`"""
    loss = 0
    for idx, (features, labels) in enumerate(train_data):

        predictions = model(features)
        loss += loss_f(predictions, labels.type(torch.LongTensor))

    loss /= idx + 1
    return loss


def loss_classifier(predictions, labels):

    criterion = nn.CrossEntropyLoss()
    return criterion(predictions, labels.type(torch.LongTensor))


def n_params(model):
    """return the number of parameters in the model"""
    n_params = sum(
        [
            np.prod([tensor.size()[k] for k in range(len(tensor.size()))])
            for tensor in list(model.parameters())
        ]
    )
    return n_params


def difference_models_norm_2(model_1, model_2):
    """Return the norm 2 difference between the two model parameters"""
    tensor_1 = list(model_1.parameters())
    tensor_2 = list(model_2.parameters())
    norm = sum(
        [
            torch.sum((tensor_1[i] - tensor_2[i]) ** 2)
            for i in range(len(tensor_1))
        ]
    )
    return norm

def local_learning(model, mu: float, optimizer, train_data, n_SGD: int, loss_f):
    model_0 = deepcopy(model)
    for _ in range(n_SGD):
        features, labels = next(iter(train_data))
        optimizer.zero_grad()
        predictions = model(features)
        batch_loss = loss_f(predictions, labels.type(torch.LongTensor))
        batch_loss += mu / 2 * difference_models_norm_2(model, model_0)
        batch_loss.backward()
        optimizer.step()


def save_pkl(dictionnary, directory, file_name):
    """Save the dictionnary in the directory under the file_name with pickle"""
    with open(f"saved_exp_info/{directory}/{file_name}.pkl", "wb") as output:
        pickle.dump(dictionnary, output)
        
def save_pkl_nodir(dictionnary, file_name):
    """Save the dictionnary in the directory under the file_name with pickle"""
    with open(f"saved_exp_info/{save_date}/{file_name}.pkl", "wb") as output:
        pickle.dump(dictionnary, output)
        
        
def get_flat_gradient(full_gradient):
    grad_1 = np.array([])
    for i_component in full_gradient:
        x = np.array(i_component)
       # print(x.shape)
        grad_1 = np.concatenate((grad_1, x.flatten()), axis=0)
    return grad_1


def get_flat_gradient_fromModel(local_model):
    flat_gradient = np.array([])
    list_params = list(local_model.parameters())
    for i_layer in list_params:
        x = np.array(i_layer.detach())
       # print(x.shape)
        flat_gradient = np.concatenate((flat_gradient, x.flatten()), axis=0)
        
    return flat_gradient