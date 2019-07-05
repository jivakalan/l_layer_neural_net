# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:08:23 2019

@author: jkalan
"""

import numpy as np 
from sklearn.datasets import load_breast_cancer

dataset= load_breast_cancer()

X = dataset['data']
Y = dataset['target']


##applies sigmoid function to input
def sigmoid(x):
    #returns s - the sigmoid of input x
    s = 1/(1+np.exp(-x))
    return s


##initalize W and B for an L layer neural network
def initialize_params(nn_dims):
   
    ##arguments: nn_dims, a list with the dimensions of each layer in the neural network
    
    ##returns: params, a dictionary containing parameters W and B for each layer
    #           W_l of shape ( nn_dims[l),nn_dims(l-1) )  
    #           B = (nn_dims[l], 1)
    
    parameters={}
    for l in range(1,len(nn_dims)):
        parameters['W'+str(l)] = np.random.randn( nn_dims[l],nn_dims[l-1] ) *0.01
        parameters['B'+str(l)] = np.zeros( ( nn_dims[l],1) )

    return parameters


def forward_prop(parameters, activation):
    
    #arguments: parameters, a dictionary with weights and biases for each layer
    #           activation, either ReLu or tanH or Sigmoid
    
    #returns: cache, a dictionary with 
    A=X.T
    cache = {}
    for l in range(1,len(nn_dims)):
        A_prev = A
        Z = np.dot(parameters['W'+str(l)],A_prev)+parameters['B'+str(l)]


        if activation == 'sigmoid':
            A = sigmoid(Z)
            
        if activation == 'relu':
            A = np.maximum(0,Z)
            
        if activation == 'tanh':
            A = np.tanh(Z)
            
        cache['Z'+str(l)]=Z
        cache['A'+str(l)]=A
        
    cache['A'+str(len(parameters)//2)] = sigmoid(cache['Z'+str(len(parameters)//2)])
     
    return cache   

def compute_loss(cache,Y):
    A= cache['A'+str(len(parameters)//2)]
    
    cost = -(1/m) * np.sum(np.multiply(Y,np.log(A)) + (1-Y)*(np.log(1-A)) )
    
    return cost #dA, cost


def backward_prop(cache, activation):
    #backward prop
    
    #arguments: cache, a dictionary with activations and outputs
    #           dA,
    #returns: update, a dictionary with update factors for weights and biases
    
    A= cache['A'+str(len(parameters)//2)]
    cache['A0'] = X.T
    dA = - (np.divide(Y, A)) + np.divide(1-Y, 1-A)
    update ={}
    for l in range(len(parameters)//2,0,-1):
        if activation == 'tanh':
            dZ = dA * np.tanh(cache['Z'+str(l)])
        if activation == 'sigmoid':
            dZ = dA * sigmoid(cache['Z'+str(l)])
        if activation == 'relu':
            dZ + dA * np.maximum(0,cache['Z'+str(l)])
            
        dW = (1/m) * np.dot(dZ, cache['A'+str(l-1)].T )
        dB = (1/m) * np.sum(dZ, axis = 1, keepdims = True)
        
        update['dZ'+str(l)]=dZ
        update['dW'+str(l)]=dW
        update['dB'+str(l)]=dB
    
    return update
        

def update_params(update, parameters):   
    #update parameters
    
    for l in range(len(parameters)//2,0,-1):
        parameters['W'+str(l)] = parameters['W'+str(l)] - learning_rate* update['dW'+str(l)]
        parameters['B'+str(l)] = parameters['B'+str(l)] - learning_rate* update['dB'+str(l)]

#5 layer neural network; 
    #n0=30
    # n1=3, 
    # n2=5,
    # n4=2, 
    # n5=1 sigmoid

nn_dims = [30,3,5,2,1]
learning_rate = 0.01
m=X.shape[0]
parameters = initialize_params(nn_dims)
#parameters = initialize_params(nn_dims)
#cache = forward_prop(parameters, activation='tanh')
#cost = compute_loss(cache,Y)
#update = backward_prop(cache, activation='tanh') 
#update_params(update,parameters)
c=1
for n in range(1,500):
    c+=1 
    cache = forward_prop(parameters, activation='tanh')
    cost = compute_loss(cache,Y)
    update = backward_prop(cache, activation='tanh') 
    update_params(update,parameters)
    print(c,cost)
    
    



#for iterations 1 to 100:
 #   do NN stuff



