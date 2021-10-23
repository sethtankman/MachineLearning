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
def PlotAccuracy(training, test, T):
    x = [0 for z in range(T)]
    for i in range(T):
        x[i] = i
    #print(arr)
    # print(errors)
    plt.plot(x, training, 'g', label='training Data')
    plt.plot(x, test, 'b', label='test Data')
    plt.legend()
    plt.xlabel("T Value")
    plt.ylabel("% Accuracy")
    plt.show()
    return 0


def BaggedLearningTrees(baggs, Set):
    finalBaggedLearners = [None for i in range(len(baggs))]
    bagNum = 0
    for bag in baggs:
        prediction = [0 for k in range(len(Set.index))]
        for tree in bag:
            for index, row in Set.iterrows():
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
        index = 0
        # print("Prediction: ", prediction)
        finalPredictions = [0 for l in range(len(prediction))]
        for label in prediction:
            finalPredictions[index] = np.sign(label)
            index += 1
        finalBaggedLearners[bagNum] = finalPredictions
        bagNum += 1
    return  finalBaggedLearners


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
    T = 20 # Should be 500
    print("DEPTH=", depth)
    # print(initWeights)

    #S = S.head(30)
    #testData = testData.head(30)

    trainingAcc = [0 for k in range(T)]
    testAcc = [0 for k in range(T)]
    reps = 20 # Should be 100
    baggs = [None for k in range(reps)]
    predictionTraining = None
    predictionTest = None
    for iteration in range(reps):
        print("iter: ", iteration)
        baggs[iteration] = []
        for t in range(T):
            print("t: ", t)
            randomIndexes = random.choices(population=S.index.tolist(), k=200) # k = 1000
            mPrime = S.iloc[randomIndexes]
            baggs[iteration].append(ID3(mPrime, Attributes, Labels, depth, "Information Gain", False))

    groundTruth = S['label'].tolist()
    # Compute bias term
    ## Pick first tree in each baggs
    hundredTrees = [baggs[i][0] for i in range(reps)]
    ## Compute the predictions of those first 100 trees
    predictions = [None for i in range(reps)]
    total = [0 for i in range(5000)]
    for t in range(reps): # where reps should be 100
        trainingAcc[t], predictions[t] = TestTree(hundredTrees[t], S, "Information Gain", t)
        total = np.add(total, predictions[t])
    ## Take the average
    average = np.divide(total, reps)
    ## subtract the ground truth (actual) label
    ## take the square
    bias = (average - groundTruth)**2
    # Compute Variance
    ## s^2 = 1/(n-1) * sum_1^n(x_i - m)^2
    ### n = reps, m = mean (or average[i]), x = predictions
    sum = 0
    for t in range(reps):
        sum += predictions[t] - average
    variance = np.sqrt(np.divide((sum**2), (len(S.index)-1)))
    singleBias = np.average(bias)
    singleVariance = np.average(variance)
    singleGSE = singleBias + singleVariance
    print("100 Bias: ", singleBias, "\n100 Variance: ", singleVariance, "\n100 GSE: ", singleGSE)
    tic1 = time.perf_counter()
    print("100 Trees processing Time: ", str(tic1 - tic0))
    # Get final estimates of bias and variance
    ## Do this by repeating above process for all test examples using the actual bagged learning trees.
    blts = BaggedLearningTrees(baggs, S)
    total1 = [0 for i in range(5000)]
    for t in range(reps):
        total1 = np.add(total, blts[t])
    average1 = np.divide(total1, reps)
    bias1 = (average1 - groundTruth) ** 2
    sum1 = 0
    for t in range(reps):
        sum1 += blts[t] - average
    variance1 = np.sqrt(np.divide((sum1 ** 2), (len(S.index) - 1)))
    ## take the average of the bias and variance values.
    bias1 = np.average(bias1)
    variance1 = np.average(variance1)
    # Get general squared error by adding them together
    generalSquarredError = bias1 + variance1
    print("Final Bias: ", bias1, "\nFinal Variance: ", variance1, "\nFinal GSE: ", generalSquarredError)

    toc = time.perf_counter()
    print("Processing Time: ", str(toc-tic0))