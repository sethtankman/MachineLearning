import time
import pandas as pd
import numpy as np
import random

from scipy.optimize import minimize

count = 0
C_i = 0
labels = None

def Constraint(alpha):
    return C_i - alpha

def Constraint1(alpha):
    return alpha

def Constraint2(alpha):
    return np.dot(alpha, labels)

# i and j iterate over the entire dataset
# y are the labels
# x are the attribute values
# alpha are the legrange multipliers for each example
def Dual(x_0, atr, labels):
    # 0.5 * sum_i sum_j y_i y_j alpha_i alpha_j x_i^T x_j - sum_i alpha_i
    # print("Dual")
    sum = 0
    y_j = labels
    x_j = atr
    alpha_j = x_0
    for i in range(len(atr)):
        y_i = labels.iloc[i]
        x_i = atr.iloc[i]
        alpha_i = x_0[i]
        # for j in range(5):
        #     y_j = labels.iloc[j]
        #     x_j = atr.iloc[j]
        #     alpha_j = x_0[j]
        #     dot = np.dot(x_i, x_j)
        #     sum += y_i * y_j * alpha_i * alpha_j * np.dot(x_i, x_j)

        sum += np.sum(y_i * y_j * alpha_i *alpha_j * x_j.dot(x_i))

    sumAlpha = np.sum(x_0)
    retVal = 0.5 * sum - sumAlpha
    return retVal


if __name__ == '__main__':
    tic0 = time.perf_counter()
    Attributes = {'variance': 0, 'skewness': 0, 'curtosis': 0, 'entropy': 0}
    prefixes = list(Attributes.keys())
    prefixes.append('label')
    S = pd.read_csv("./bank-note/train.csv", header=None, names=prefixes)
    S['label'] = S['label'].map({0: -1, 1: 1})
    S.insert(4, 'bias', [1 for i in range(len(S.index))])
    testData = pd.read_csv("./bank-note/test.csv", header=None, names=prefixes)
    testData['label'] = testData['label'].map({0: -1, 1: 1})
    testData.insert(4, 'bias', [1 for i in range(len(testData.index))])

    T = 100
    C = [100/873, 500/873, 700/873]
    gamma = [0.1,0.5,1.0,5.0,100.0]

    atr = S.loc[:, S.columns != 'label']
    labels = S['label']
    x_0 = np.zeros(shape=(len(atr.index),))
    con1 = {'type': 'ineq', 'fun': Constraint}
    con2 = {'type': 'ineq', 'fun': Constraint1}
    con3 = {'type': 'eq', 'fun': Constraint2}
    cons = ([con1, con2, con3])

    # np.outer you can get the objective function out of 4 lines.
    # np.broadcast for subtractions, reshape, instantiate the matrix you put out with np.empty
    x = np.array(atr)
    b = np.broadcast(x, x)
    out = np.empty(b.shape)
    out.flat = [u-v for (u,v) in b]
    out

    for i in range(len(C)):
        print("\n\nC=", C[i], "\n\n")
        C_i = C[i]
        res = minimize(Dual, x_0, args=(atr, labels), method="SLSQP", constraints=cons)
        mult = pd.DataFrame([res.x * labels] * 5)
        preW = pd.DataFrame(mult.T.values * atr.values, columns= atr.columns, index=atr.index)
        w = preW.sum(axis=0)
        # wtx = pd.DataFrame(w.values * atr, columns=atr.columns, index=atr.index)
        b= 0
        for i in range(len(labels.index)):
            b += labels.iloc[i] - np.dot(w, atr.iloc[i])
        b /= len(atr.index)
        print(res)
        print("W: ", w)
        print("B: ", b)

    toc = time.perf_counter()
    print("Processing Time: ", str(toc-tic0))