import time
import pandas as pd
import numpy as np
import random
import torch

class neuron:
    def __init__(self, name):
        self.name = name
        self.parents = {} # dictionary of name,weight pairs
        self.children = {}

    def addParent(self, _neuron, weight):
        self.parents[_neuron] = weight
        _neuron.children[self] = weight

    def addChild(self, _neuron, weight):
        self.children[_neuron] = weight
        _neuron.parents[self] = weight

def sigmoid(x):
    return 1/(1 + torch.exp(-x))

def makePrediction(input_vector, weights, bias):
    s = torch.dot(input_vector, weights) + bias
    return sigmoid(s)

# make a three-layered neural network based on the number of attributes
# Weights are accessed by weights[height][z-value][weight-vector]
def makeWeights(n_atr):
    weights = [None for a in range(3)]
    for i in range(2): # Can change range to add more layers to NN.
        weights[i] = [None for b in range(n_atr)]
        for j in range(n_atr):
            weights[i][j] = [None for c in range(n_atr+1)]
            for k in range(n_atr+1):
                weights[i][j][k] = torch.rand(1,requires_grad=True, dtype=torch.float64)
    weights[2] = torch.ones(n_atr+1, dtype=torch.float64)
    for l in range(n_atr+1):
        weights[2][l] = torch.rand(1,requires_grad=True, dtype=torch.float64)
    return weights

def my_dot(w, x):
    sum = torch.tensor([0.0],dtype= torch.float64, requires_grad=True)
    sum.retain_grad()
    for i in range(len(x)):
        w[i].retain_grad()
        x[i].retain_grad()
        sum = sum + w[i] * x[i]
    return sum

# X is list of attribute values with bias as the first element.
def forward_prop(X, weights):
    num_atr = len(X)
    #Z1 = torch.ones(num_atr, requires_grad=True, dtype=torch.float64) # the z values for layer 1
    #Z2 = torch.ones(num_atr, requires_grad=True, dtype=torch.float64)
    Z1 = [torch.tensor(1.0,dtype= torch.float64, requires_grad=True) for a in range(num_atr)]
    Z2 = [torch.tensor(1.0,dtype= torch.float64, requires_grad=True) for b in range(num_atr)]
    for i in range(1, num_atr):
        Z1[i] = my_dot(weights[0][i-1], X)
        Z1[i].retain_grad()
        Z1[i] = sigmoid(Z1[i])
    for j in range(1, num_atr):
        Z2[j] = my_dot(weights[1][j-1], Z1)
        Z2[j].retain_grad()
        Z2[j] = sigmoid(Z2[j])
    y = my_dot(weights[2], Z2)
    y.retain_grad()
    return y

def get_cost(y, y0):
    return 1/2*(y - y0)**2

if __name__ == '__main__':
    tic0 = time.perf_counter()
    Attributes = {'variance': 0, 'skewness': 0, 'curtosis': 0, 'entropy': 0}
    prefixes = list(Attributes.keys())
    prefixes.append('label')
    S = pd.read_csv("./bank-note/train.csv", header=None, names=prefixes)
    S.insert(0, 'bias', [1 for i in range(len(S.index))])
    testData = pd.read_csv("./bank-note/test.csv", header=None, names=prefixes)
    testData.insert(0, 'bias', [1 for i in range(len(testData.index))])
    x_0 = S.iloc[0].to_numpy()
    y_0 = x_0[-1] # save the label
    x_0 = torch.tensor(x_0[0:-1], requires_grad=True) # remove the label

    weights = makeWeights(len(Attributes))
    y = forward_prop(x_0, weights)
    L = get_cost(y, y_0)
    # y = neuron('y')
    # z_0_2 = neuron('z_0_2')
    # z_1_2 = neuron('z_1_2')
    # z_2_2 = neuron('z_2_2')
    # z_0_1 = neuron('z_0_1')
    # z_1_1 = neuron('z_1_1')
    # z_2_1 = neuron('z_2_1')
    # x_0 = neuron('x_0')
    # x_1 = neuron('x_1')
    # x_2 = neuron('x_2')
    #
    # y.addChild(z_0_2, -1)
    # y.addChild(z_1_2, 2)
    # y.addChild(z_2_2, -1.5)
    # z_1_2.addChild(z_0_1, -1)
    # z_1_2.addChild(z_1_1, -2)
    # z_1_2.addChild(z_2_1, -3)
    # z_2_2.addChild(z_0_1, 1)
    # z_2_2.addChild(z_1_1, 2)
    # z_2_2.addChild(z_2_1, 3)
    # z_1_1.addChild(x_0, -1)
    # z_1_1.addChild(x_1, -2)
    # z_1_1.addChild(x_2, -3)
    # z_2_1.addChild(x_0, 1)
    # z_2_1.addChild(x_1, 2)
    # z_2_1.addChild(x_2, 3)

    L.backward()
    print(y.grad)
    for level in range(3):
        for row in range(len(Attributes)):
            for col in range(len(Attributes)+1):
                inspect = weights[level][row][col]
                print("W_",row,col,"^",level,": ",weights[level][row][col].grad)



    toc = time.perf_counter()
    print("Processing Time: ", str(toc-tic0))