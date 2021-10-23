import math
import pandas as pd
import numpy as np
import time
import random

from matplotlib import pyplot as plt

def StochasticGradientDescent(r, S, Attributes):
    wVector = [0 for i in range(len(Attributes.keys()))]
    totalError = 5
    threshold = 0.000001
    t=0
    cost = []
    # m = number of examples, i = current iteration on the way to m. j is each element of the d items in the weight vector
    y = S['label'].tolist()
    x = S[S.columns.difference(['label'])]
    while totalError >= threshold:
        wPrime = [0 for i in range(len(Attributes.keys()))]
        index = random.sample(x.index.tolist(), 1)[0]
        row = x.iloc[index]
        for j in range(len(wVector)):
            wPrime[j] = wVector[j] + r * (y[index] - np.dot(wVector,row.values)) * row.values[j]
        npWPrime = np.array(wPrime)
        print(totalError)
        sum = 0
        for index, row in x.iterrows():
            sum += np.power(y[index] - np.dot(npWPrime,row.values), 2)
        cost.append(0.5 * sum)
        if len(cost) > 2:
            totalError = abs(cost[len(cost)-1] - cost[len(cost)-2])
        print(totalError)
        wVector = wPrime
        t += 1

    print("Iterations: ", t)
    return cost, wVector

if __name__ == '__main__':
    tic0 = time.perf_counter()
    Labels = [1, -1]
    # attrubute: (isNumeric, list)
    Attributes = {'Cement': 0, 'Slag': 0, 'Fly ash': 0, 'Water': 0, 'SP': 0, 'Coarse Aggr': 0, 'Fine Aggr': 0}
    prefixes = list(Attributes.keys())
    prefixes.append('label')
    S = pd.read_csv("./concrete/train.csv", header=None, names=prefixes)
    testData = pd.read_csv("./concrete/test.csv", header=None, names=prefixes)

    # Implement batch gradient descent
    r = 0.015
    cost, wVector = StochasticGradientDescent(r, S, Attributes)

    print("Learning Rate: ", r, "\nWeightVector: ", wVector)

    y = testData['label'].tolist()
    x = testData[testData.columns.difference(['label'])]
    sum = 0
    for index, row in x.iterrows():
        sum += np.power(y[index] - np.dot(wVector, row.values), 2)
    print("Cost on test data: ", 0.5 * sum)

    toc = time.perf_counter()
    print("Processing Time: ", str(toc-tic0))

    x = [0 for z in range(len(cost))]
    for i in range(len(cost)):
        x[i] = i
    plt.plot(x, cost, 'tomato', label='cost')
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.show()
