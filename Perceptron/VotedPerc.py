import time
import pandas as pd
import numpy as np
import random
import os

# Returns a weight vector
def VotedPerceptron(S, T, r):
    #Initialize Weights
    weightVector = [[0 for i in range(5)]]
    m = 0
    C_m = [0]
    for t in range(T):
        #Shuffle the data
        randomIndexes = random.choices(population=S.index.tolist(), k=len(S.index))
        mPrime = S.iloc[randomIndexes]
        # For each example
        for index, row in mPrime.iterrows():
            y_i = row['label']
            x_i = row[0:-1].values
            if y_i*np.dot(weightVector[m],x_i) <= 0:
                weightVector.append(weightVector[m] + r*y_i*x_i)
                m = m+1
                C_m.append(1)
            else:
                C_m[m] += 1

    votedWeights = []
    index = 0
    for weightVec in weightVector:
        votedWeights.append((weightVec, C_m[index]))
        index += 1
    return votedWeights

# Returns an average prediction error
def GetAvgPredictError(weightVectors, TestData):
    numExamples = len(TestData.index)
    error = 0
    for index, row in TestData.iterrows():
        y_i = row['label']
        x_i = row[0:-1].values
        prediction = 0
        for i in range(len(weightVectors)):
            prediction += weightVectors[i][1] * np.sign(np.dot(weightVectors[i][0], x_i))
        prediction = np.sign(prediction)
        if prediction != y_i:
            error +=1
    return error/numExamples

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

    T = 10
    r=0.01
    print("Learning Rate: ", r)

    weightVectors = VotedPerceptron(S, T, r)
    avgPredictError = GetAvgPredictError(weightVectors, testData)
    csv_wVectors = []
    for entry in weightVectors:
        csv_entry = None
        if type(entry[0]) is list:
            csv_entry = entry[0]
            csv_entry.append(entry[1])
        else:
            csv_entry = entry[0].tolist()
            csv_entry.append(entry[1])
        csv_wVectors.append(csv_entry)
    df = pd.DataFrame(csv_wVectors)
    df.to_csv(os.getcwd() + '/votedWeightVectors.csv', index=False)
    print("Average Prediction Error: ", avgPredictError)

    toc = time.perf_counter()
    print("Processing Time: ", str(toc-tic0))