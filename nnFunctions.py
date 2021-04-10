import numpy as np
from scipy.optimize import minimize
from math import sqrt
import pickle
'''
You need to modify the functions except for initializeWeights() 
'''

def initializeWeights(n_in, n_out):
    '''
    initializeWeights return the random weights for Neural Network given the
    number of node in the input layer and output layer

    Input:
    n_in: number of nodes of the input layer
    n_out: number of nodes of the output layer

    Output:
    W: matrix of random initial weights with size (n_out x (n_in + 1))
    '''
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    '''
    Notice that z can be a scalar, a vector or a matrix
    return the sigmoid of input z (same dimensions as z)
    '''
    # remove the next line and replace it with your code
    if isinstance(z, (float, int)):
        z = 1/(1+np.exp(-z))
    else:
        shape = z.shape
        z = z.flatten()
        z = np.array(list(map(lambda val: 1/(1+np.exp(-val)), z)))
        z = z.reshape(shape)
    return z 

def nnObjFunction(params, *args):
    '''
    % nnObjFunction computes the value of objective function (cross-entropy
    % with regularization) given the weights and the training data and lambda
    % - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices W1 (weights of connections from
    %     input layer to hidden layer) and W2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of nodes in input layer (not including the bias node)
    % n_hidden: number of nodes in hidden layer (not including the bias node)
    % n_class: number of nodes in output layer (number of classes in
    %     classification problem
    % train_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of the corresponding instance 
    % train_label: the vector of true labels of training instances. Each entry
    %     in the vector represents the truee label of its corresponding training instance.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector (not a matrix) of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.
    '''
    # do not remove the next 5 lines
    n_input, n_hidden, n_class, train_data, train_label, lambdaval = args
    # First reshape 'params' vector into 2 matrices of weights W1 and W2

    W1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    W2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    # remove the next two lines and replace them with your code 
    ####################compute objective function########################
    train_label = np.array(list(map(int, train_label)))
    bias0 = np.ones((train_data.shape[0], 1))
    X = np.concatenate((train_data, bias0), axis=1)
    l1_input = X @ W1.T
    l1_output = sigmoid(l1_input)
    bias1 = np.ones((l1_output.shape[0], 1))
    l1_output = np.concatenate((l1_output, bias1), axis=1)
    l2_input = l1_output @ W2.T
    l2_output = sigmoid(l2_input)
    one_of_k = np.zeros((train_label.size, train_label.max()+1), dtype="int64")
    one_of_k[np.arange(train_label.size),train_label] = 1

    loss=0
    for i in range(train_data.shape[0]):
        for l in range(n_class):
            yil = one_of_k[i][l]
            oil = l2_output[i][l]

            loss+=(yil*np.log(oil))
            loss+=((1-yil)*np.log(1-oil))
    loss*=(-1/train_data.shape[0])

    reg = lambdaval*(np.sum(W1.reshape(-1)**2) + np.sum(W2.reshape(-1)**2))/(2*train_data.shape[0])

    obj_val = loss+reg

    ####################compute gradient of objective function########################
    w1_grad = []
    w2_grad = []
    for i in range(train_data.shape[0]):
        ############gradient w.r.t. W2################
        delta = (l2_output[i] - one_of_k[i]).reshape(1,-1)
        z = (l1_output[i]).reshape(1,-1)
        
        w2_grad.append((delta.T @ z).flatten())

        ############gradient w.r.t. W1################
        z = (l1_output[:,:-1][i]).reshape(1,-1)
        z = z*(1-z)
        phi = delta @ W2[:,:-1]
        phi = z*phi
        Xp = (X[i]).reshape(1,-1)
        w1_grad.append((phi.T @ Xp).flatten())

    w1_grad=(np.sum(w1_grad, axis=0)+lambdaval*W1.flatten())/train_data.shape[0]
    w2_grad=(np.sum(w2_grad, axis=0)+lambdaval*W2.flatten())/train_data.shape[0]
    
    obj_grad = np.hstack([w1_grad, w2_grad])
    return (obj_val,obj_grad)

def nnPredict(W1, W2, data):
    '''
    % nnPredict predicts the label of data given the parameter W1, W2 of Neural
    % Network.

    % Input:
    % W1: matrix of weights for hidden layer units
    % W2: matrix of weights for output layer units
    % data: matrix of data. Each row of this matrix represents the feature vector for the corresponding data instance

    % Output:
    % label: a column vector of predicted labels
    '''
    # remove the next line and replace it with your code
    # Your code here

    labels = []
    bias = np.ones([data.shape[0]])
    data = np.column_stack([data, bias])
    hidden_layer = sigmoid(np.dot(data, W1.T))

    bias1 = np.ones([hidden_layer.shape[0]])
    data1 = np.column_stack([hidden_layer,bias1])
    outer_layer = sigmoid(np.dot(data1, W2.T))
    labels = np.argmax(outer_layer, axis = 1)
    return labels