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

def Test(weights, S):
    error = 0
    numExamples = len(S.index)
    for index, row in S.iterrows():
        y_i = row['label']
        x_i = row[0:-1].values
        if np.sign(np.dot(weights, x_i)) != y_i:
            error += 1
    return error / numExamples

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

    atr = S.loc[:, S.columns != 'label']
    labels = S['label']

    AllWeights = [[-0.9429264, -0.6514918, -0.7337219, -0.04102187,0],[-1.563939, -1.014053, -1.180652, -0.1565176, 0],[-2.042621,-1.280140,-1.513290,-0.2483647,0]]
    print("Weights obtained from Dual.py: ", AllWeights)

    for LearnedWeights in AllWeights:
        print("Training Error: ", Test(LearnedWeights, S))
        print("Test Error: ", Test(LearnedWeights, testData))

    toc = time.perf_counter()
    print("Processing Time: ", str(toc-tic0))