import time
import pandas as pd
import numpy as np
import random


def SGD(T, C, LR, S):
    w_0 = [0 for i in range(len(S.columns)-1)]
    weights = [0 for i in range(len(S.columns)-1)]
    a = 2
    N = len(S.index)
    for t in range(T):
        LR_t = LR /(1 + t)
        # LR_t = LR / (1 + LR/a*t)
        # Shuffle the data
        randomIndexes = random.choices(population=S.index.tolist(), k=len(S.index))
        mPrime = S.iloc[randomIndexes]
        # For each example
        for index, row in mPrime.iterrows():
            y = row['label']
            x_i = row[0:-1].values
            ywx = y * np.dot(weights, x_i)
            if ywx <= 1:
                weights = weights - np.multiply(LR_t, w_0) + LR_t * C * N * y * x_i #Check for 0 bias term?
            else:
                w_0 = np.multiply((1-LR_t),w_0)
        return weights

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
    learningRate = 0.001
    print("Learning Rate: ", learningRate)
    print("a = 2")

    for i in range(len(C)):
        print("C = ", C[i])
        LearnedWeights = SGD(T, C[i], learningRate, S)
        print("Training Error: ", Test(LearnedWeights, S))
        print("Test Error: ", Test(LearnedWeights, testData))


    toc = time.perf_counter()
    print("Processing Time: ", str(toc-tic0))