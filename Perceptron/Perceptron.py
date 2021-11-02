import time
import pandas as pd
import numpy as np
import random

# Returns a weight vector
def Perceptron(S, T, r):
    #Initialize Weights
    weightVector = [0 for i in range(4)]
    for t in range(T):
        #Shuffle the data
        randomIndexes = random.choices(population=S.index.tolist(), k=len(S.index))
        mPrime = S.iloc[randomIndexes]
        # For each example
        for index, row in mPrime.iterrows():
            y_i = row['label']
            x_i = row[0:-1].values
            if y_i*np.dot(weightVector,x_i) <= 0:
                weightVector = weightVector + r*y_i*x_i

    return weightVector

# Returns an average prediction error
def GetAvgPredictError(weightVector, TestData):
    numExamples = len(TestData.index)
    error = 0
    for index, row in TestData.iterrows():
        y_i = row['label']
        x_i = row[0:-1].values
        if np.sign(np.dot(weightVector,x_i)) != y_i:
            error +=1
    return error/numExamples

if __name__ == '__main__':
    tic0 = time.perf_counter()
    Attributes = {'variance': 0, 'skewness': 0, 'curtosis': 0, 'entropy': 0}
    prefixes = list(Attributes.keys())
    prefixes.append('label')
    S = pd.read_csv("./bank-note/train.csv", header=None, names=prefixes)
    S['label'] = S['label'].map({0: -1, 1: 1})
    testData = pd.read_csv("./bank-note/test.csv", header=None, names=prefixes)
    testData['label'] = testData['label'].map({0: -1, 1: 1})

    T = 10
    r=0.01
    print("Learning Rate: ", r)

    weightVector = Perceptron(S, T, r)
    avgPredictError = GetAvgPredictError(weightVector, testData)
    print("Weight Vector: ", weightVector)
    print("Average Prediction Error: ", avgPredictError)

    toc = time.perf_counter()
    print("Processing Time: ", str(toc-tic0))