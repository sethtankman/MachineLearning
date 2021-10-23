import math
import pandas as pd
import numpy as np
import time
import random

from matplotlib import pyplot as plt

class Node:
    def __init__(self, branches, label, atr):
        self.branches = branches
        self.label = label
        self.attribute = atr

    def printNodes(self, finalString, numLevels):
        finalString += str(self.label) + "\n"
        if len(self.branches.keys()) == 0:
            return finalString
        finalString += "splitting on: " + self.attribute + "\n"
        for key in self.branches.keys():
            # print("#", key, "=", end= "")
            finalString += "#"*(numLevels+1) + key + "="
            finalString += self.branches[key].printNodes("", numLevels+1)
        return finalString

# Finds the list of all labels and the most common label
def GetTheDeets(S):
    allLabels = {}
    # count the number of examples that have each labels and store that in allLabels
    for index, row in S.iterrows():
        if row['label'] in allLabels.keys():
            allLabels[row['label']] += 1
        else:
            allLabels[row['label']] = 1
    mostCommonLabel = None
    # print("All Labels:", allLabels)
    for eachLabel in allLabels:
        # print("This label: ", eachLabel)
        if mostCommonLabel is None or allLabels[eachLabel] > mostCommonLabel[1]:
            mostCommonLabel = (eachLabel, allLabels[eachLabel])
    return allLabels, mostCommonLabel[0]


def Subset(S, A, value):
    return S.loc[S[A] == value]
    # return S where A = value

# divide number of examples with attributes by total number of attrubutes
# multiply that by -pos log(pos) - neg log(neg)
def Entropy(Set, Labels):
    totalEntropy = 0
    # debugFrac = 0
    for label in Labels:
        # print("Label: ", label, "Numerator: ", len(S.loc[S['label'] == label]), "Denominator: ", len(S))
        matchingLabel = Set.loc[Set['label'] == label]
        #print(len(matchingLabel))
        if len(matchingLabel) == 0:
            continue
        frac = len(matchingLabel.index) / len(Set.index)
        # debugFrac += frac
        if frac == 0:
            continue
        #elif frac >= 1 or frac < 0:
        #    print("Frac: ", frac)
        # print(-frac)
        # print(math.log(frac))
        # print("Sub-entropy: ", -frac * math.log(frac))
        totalEntropy += -frac * math.log(frac, 2)
        # totalEntropy += - len(S.loc[S['label'] == label]) / len(S) * math.log(len(S.loc[S['label'] == label]) / len(S))
    #print("Fraction total is: ", debugFrac)
    return totalEntropy

# returns the attribute that best splits the set S
def BestSplit(S, Attributes, Labels, chooser, weCareAboutUnknowns):
    gains = {}
    initEntropy = Entropy(S, Labels)
    #print("InitEntropy: ", initEntropy)
    allExamples = len(S.index)
    # print("This should be 1: ", allExamples)
    # print("Attributes:", Attributes.keys())
    for attribute in Attributes.keys(): # We check each attribute's information gain
        # print("Attribute: ", attribute)
        totalGains = initEntropy
        # debugScale = 0
        # print("InitialTotal: ", totalGains)
        if Attributes[attribute][0] == 0: # Non-numeric attribute values
            for value in Attributes[attribute][1]: # for each value of the attribute
                # if weCareAboutUnknowns and value == 'unknown':
                #     otherVals = Attributes[attribute][1].copy()
                #     #print("OTHER: ", otherVals)
                #     otherVals.remove(value)
                #     mostCommonAtrVal = ("", -1)
                #     for contestVal in otherVals:
                #         contestAtrNum = len(S.loc[S[attribute] == contestVal])
                #         if contestAtrNum > mostCommonAtrVal[1]:
                #             mostCommonAtrVal = (contestVal, contestAtrNum)
                #     value = mostCommonAtrVal[0]
                atrSubset = Subset(S, attribute, value)
                # print("Length of subset: ", len(atrSubset), " Length of Whole: ", allExamples)
                atrScale = len(atrSubset.index) / allExamples
                # debugScale += atrScale
                # print("Scale: ", atrScale)
                entr = Entropy(atrSubset, Labels)
                # print("Subtractor Entropy: ", entr)
                # print("-", (atrScale * entr))
                totalGains -= (atrScale * entr)
                # print("-Total: ", totalGains)
        else: # Numeric attribute values
            numericValues = S[attribute].tolist()
            median = np.median(numericValues)
            #print("numeric: ", numericValues, "\nMedian: ", median)
            atrSubset = S.loc[S[attribute] > median]
            #print(attribute, ": ", atrSubset)
            atrScale = len(atrSubset.index) / allExamples
            # debugScale += atrScale
            entr = Entropy(atrSubset, Labels)
            # print("-", (atrScale* entr))
            totalGains -= (atrScale * entr)
            # print("-Total: ", totalGains)

            atrSubset = S.loc[S[attribute] <= median]

            #print(attribute, ": ", atrSubset)
            atrScale = len(atrSubset.index) / allExamples
            # debugScale += atrScale
            entr = Entropy(atrSubset, Labels)
            # print("-", (atrScale* entr))
            totalGains -= (atrScale * entr)
            # print("-Total: ", totalGains)
        # print("Debug Scale: ", debugScale)
        gains[attribute] = totalGains
    greatestGains = None
    # print("Gains:", gains)
    for atr in gains.keys():
        if greatestGains is None or gains[atr] > greatestGains[1]:
            greatestGains = (atr, gains[atr])
    # print("Greatest: ", greatestGains)
    return greatestGains[0]


# The ID3 algorithm
# S = The set of examples
# Attributes = the set of measured attributes
# Label = The target attribute (prediction)
# returns a decision tree where each node has a label, attribute, and a list of branches.
def ID3(S, Attributes, Labels, depth, chooser, weCareAboutUnknowns):
    allLabels, mostCommonLabel = GetTheDeets(S)
    if len(list(allLabels.keys())) == 1:
        return Node({}, str(list(allLabels.keys())[0]), "")
    if len(Attributes) == 0 or depth < 1:
        return Node({}, mostCommonLabel, "")
    rootNode = Node({}, "", "")
    # print("Attributes pre BestSplit: ", Attributes)
    A = BestSplit(S, Attributes, allLabels.keys(), chooser, weCareAboutUnknowns)
    rootNode.attribute = A
    if Attributes[A][0] == 0: # For non-numeric values
        for value in Attributes[A][1]:
            # Adding a branch is done implicitly
            # print("A: ", A, "val: ", value)
            Sv = Subset(S, A, value)
            if len(Sv) == 0:
                rootNode.branches[value] = Node({}, mostCommonLabel, "")
            else:
                newAtr = Attributes.copy()
                # print("Attributes: ", Attributes)
                # print("Removing: ", A)
                newAtr.pop(A)
                rootNode.branches[value] = ID3(Sv, newAtr, Labels, depth - 1, chooser, weCareAboutUnknowns)
    else: #For numeric values
        numericValues = S[A].tolist()
        median = np.median(numericValues)
        Sv = S.loc[S[A] > median]
        if len(Sv) == 0:
            rootNode.branches[">"+str(median)] = Node({}, mostCommonLabel, "")
        else:
            newAtr = Attributes.copy()
            # print("Attributes: ", Attributes)
            # print("Removing: ", A)
            newAtr.pop(A)
            rootNode.branches[">"+str(median)] = ID3(Sv, newAtr, Labels, depth - 1, chooser, weCareAboutUnknowns)
        Sv = S.loc[S[A] <= median]
        if len(Sv) == 0:
            rootNode.branches["<"+str(median)] = Node({}, mostCommonLabel, "")
        else:
            newAtr = Attributes.copy()
            newAtr.pop(A)
            rootNode.branches["<"+str(median)] = ID3(Sv, newAtr, Labels, depth - 1, chooser, weCareAboutUnknowns)
    return rootNode

# def TestTrees(trees, testData, chooser, T):
#     prediction = [0 for k in range(len(testData.index))]
#     for c in range(len(trees)):
#         tree = trees[c]
#         for index, row in testData.iterrows():
#             currentNode = tree
#             while currentNode.attribute != "":
#                 atrValue = row[currentNode.attribute]
#                 # print(currentNode.attribute)
#                 # print(currentNode.branches)
#                 for branch in currentNode.branches.keys():
#                     if branch[0] == '>':
#                         if atrValue > int(branch[1:-2]):
#                             currentNode = currentNode.branches[branch]
#                             break
#                     elif branch[0] == '<':
#                         if atrValue <= int(branch[1:-2]):
#                             currentNode = currentNode.branches[branch]
#                             break
#                     elif branch == atrValue:
#                         # print(currentNode.branches)
#                         # print(atrValue)
#                         currentNode = currentNode.branches[atrValue]
#                         break
#             prediction[index] += int(currentNode.label)
#     accurate = 0
#     index = 0
#     # print("Prediction: ", prediction)
#     for label in prediction:
#         finalPrediction = np.sign(label)
#         if finalPrediction == testData.iloc[index]['label']:
#             accurate += 1
#         #else:
#         #    print(index)
#         index += 1
#     print(chooser, ": ", accurate/index, "% accurate")
#     return accurate/index

def TestTree(tree, testData, chooser, T, prediction=None):
    if prediction == None:
        prediction = [0 for k in range(len(testData.index))]
    for index, row in testData.iterrows():
        currentNode = tree
        while currentNode.attribute != "":
            atrValue = row[currentNode.attribute]
            # print(currentNode.attribute)
            # print(currentNode.branches)
            for branch in currentNode.branches.keys():
                if branch[0] == '>':
                    if atrValue > int(branch[1:-2]):
                        currentNode = currentNode.branches[branch]
                        break
                elif branch[0] == '<':
                    if atrValue <= int(branch[1:-2]):
                        currentNode = currentNode.branches[branch]
                        break
                elif branch == atrValue:
                    # print(currentNode.branches)
                    # print(atrValue)
                    currentNode = currentNode.branches[atrValue]
                    break
        prediction[index] += int(currentNode.label)
    accurate = 0
    index = 0
    # print("Prediction: ", prediction)
    for label in prediction:
        finalPrediction = np.sign(label)
        if finalPrediction == testData.iloc[index]['label']:
            accurate += 1
        #else:
        #    print(index)
        index += 1
    # print(chooser, ": ", accurate/index, "% accurate")
    return accurate/index, prediction

# Press the green button in the gutter to run the script.
def PlotEverything(trainingErrors, testErrors, T):
    x = [0 for z in range(T)]
    for i in range(T):
        x[i] = i
    plt.plot(x, trainingErrors[0], 'tomato', label='training-2')
    plt.plot(x, testErrors[0], 'cyan', label='test-2')
    plt.plot(x, trainingErrors[1], 'red', label='training-4')
    plt.plot(x, testErrors[1], 'blue', label='test-4')
    plt.plot(x, trainingErrors[2], 'darkred', label='training-6')
    plt.plot(x, testErrors[2], 'darkblue', label='test-6')
    plt.legend()
    plt.xlabel("T Value")
    plt.ylabel("Error")
    plt.show()
    return 0


def RandTreeLearn(S, Attributes, depth):
    allLabels, mostCommonLabel = GetTheDeets(S)
    if len(list(allLabels.keys())) == 1:
        return Node({}, str(list(allLabels.keys())[0]), "")
    if len(Attributes) == 0 or depth < 1:
        return Node({}, mostCommonLabel, "")
    rootNode = Node({}, "", "")
    subsetSize = len(Attributes.keys())/2
    if subsetSize < 1:
        subsetSize = 1
    gKeys = random.sample(Attributes.keys(), int(subsetSize))
    G = {}
    for key in gKeys:
        G[key] = Attributes[key]

    # print("Attributes pre BestSplit: ", Attributes)
    A = BestSplit(S, G, allLabels.keys(), "Information Gain", False)
    rootNode.attribute = A
    if Attributes[A][0] == 0: # For non-numeric values
        for value in Attributes[A][1]:
            # Adding a branch is done implicitly
            # print("A: ", A, "val: ", value)
            Sv = Subset(S, A, value)
            if len(Sv) == 0:
                rootNode.branches[value] = Node({}, mostCommonLabel, "")
            else:
                newAtr = Attributes.copy()
                # print("Attributes: ", Attributes)
                # print("Removing: ", A)
                newAtr.pop(A)
                rootNode.branches[value] = RandTreeLearn(Sv, newAtr, depth - 1)
    else: #For numeric values
        numericValues = S[A].tolist()
        median = np.median(numericValues)
        Sv = S.loc[S[A] > median]
        if len(Sv) == 0:
            rootNode.branches[">"+str(median)] = Node({}, mostCommonLabel, "")
        else:
            newAtr = Attributes.copy()
            # print("Attributes: ", Attributes)
            # print("Removing: ", A)
            newAtr.pop(A)
            rootNode.branches[">"+str(median)] = RandTreeLearn(Sv, newAtr, depth - 1)
        Sv = S.loc[S[A] <= median]
        if len(Sv) == 0:
            rootNode.branches["<"+str(median)] = Node({}, mostCommonLabel, "")
        else:
            newAtr = Attributes.copy()
            newAtr.pop(A)
            rootNode.branches["<"+str(median)] = RandTreeLearn(Sv, newAtr, depth - 1)
    return rootNode

# Returns a list of predictions over a dataFrame using the averages of 1 to T tree predictions
# Ultimately this will be a T x len(df.index) table.
def AveragePredictions(trees, S, T):
    averagePrediction = [[0 for l in range(len(S.index))] for k in range(T)]
    prediction = [0 for k in range(len(S.index))]
    for t in range(T):
        tree = trees[t]
        for index, row in S.iterrows():
            currentNode = tree
            while currentNode.attribute != "":
                atrValue = row[currentNode.attribute]
                # print(currentNode.attribute)
                # print(currentNode.branches)
                for branch in currentNode.branches.keys():
                    if branch[0] == '>':
                        if atrValue > int(branch[1:-2]):
                            currentNode = currentNode.branches[branch]
                            break
                    elif branch[0] == '<':
                        if atrValue <= int(branch[1:-2]):
                            currentNode = currentNode.branches[branch]
                            break
                    elif branch == atrValue:
                        # print(currentNode.branches)
                        # print(atrValue)
                        currentNode = currentNode.branches[atrValue]
                        break
            prediction[index] += int(currentNode.label)
            averagePrediction[t][index] = np.sign(prediction[index])
    return averagePrediction


def GetErrors(predictions, S, testData):
    trainingErrorsAtF = [None, None, None]
    testErrorsAtF = [None, None, None]
    FIndex = 0
    for predictionAtF in predictions:
        tValue = 0
        trainErrors = [0 for k in range(len(predictionAtF))]
        testErrors = [0 for k in range(len(predictionAtF))]
        for prediction in predictionAtF:
            A = np.array(prediction)
            B = np.array(S['label'].tolist())
            C = np.array(testData['label'].tolist())
            trainErr = (A != B).sum() / len(S.index)
            testErr = (A != C).sum() / len(S.index)
            trainErrors[tValue] = trainErr
            testErrors[tValue] = testErr
            tValue += 1
        trainingErrorsAtF[FIndex] = trainErrors
        testErrorsAtF[FIndex] = testErrors
        FIndex += 1
    return trainingErrorsAtF, testErrorsAtF


if __name__ == '__main__':
    tic0 = time.perf_counter()
    Labels = [1, -1]
    # attrubute: (isNumeric, list)
    Attributes = {'age': (1, []), 'job': (0, ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
                                    "blue-collar","self-employed","retired","technician","services"]),
                  'marital': (0, ["married","divorced","single"]), 'education': (0, ["unknown","secondary","primary","tertiary"]),
                  'default': (0, ["yes","no"]), 'balance': (1, []), 'housing': (0, ["yes","no"]), 'loan': (0, ["yes","no"]),
                  'contact': (0, ["unknown","telephone","cellular"]), 'day': (1, []),
                  'month': (0, ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]),
                  'duration': (1, []), 'campaign': (1, []), 'pdays': (1, []), 'previous': (1, []),
                  'poutcome': (0, ["unknown","other","failure","success"])}
    prefixes = list(Attributes.keys())
    prefixes.append('label')
    S = pd.read_csv("../DecisionTree/bank/train.csv", header=None, names=prefixes)
    S['label'] = S['label'].map({'no': -1, 'yes': 1})
    testData = pd.read_csv("../DecisionTree/bank/test.csv", header=None, names=prefixes)
    testData['label'] = testData['label'].map({'no': -1, 'yes': 1})

    depth = 20
    T = 500 # Should be 500
    print("DEPTH=", depth)

    #S = S.head(30)
    #testData = testData.head(30)
    numFeatures = [2, 4, 6]
    trees = [None for k in range(T)]
    predictions = [None, None, None]
    for subsetSize in numFeatures:
        featureKeys = random.sample(Attributes.keys(), subsetSize)
        featureSubset = {}
        for key in featureKeys:
            featureSubset[key] = Attributes[key]
        for t in range(T):
            print("t: ", t)
            randomIndexes = random.choices(population=S.index.tolist(), k=1000) # k = 1000
            mPrime = S.iloc[randomIndexes]
            trees[t] = RandTreeLearn(mPrime, featureSubset, depth)
        predictions[int(subsetSize/2 -1)] = AveragePredictions(trees, S, T)

    trainingErrors, testErrors = GetErrors(predictions, S, testData)

    toc = time.perf_counter()
    print("Processing Time: ", str(toc-tic0))
    PlotEverything(trainingErrors, testErrors, T)