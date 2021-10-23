import math
import pandas as pd
import numpy as np
import time

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
def GetTheDeets(S, weights):
    allLabels = {}
    # count the number of examples that have each labels and store that in allLabels
    for index, row in S.iterrows():
        if row['label'] in allLabels.keys():
            allLabels[row['label']] += weights[index]
        else:
            allLabels[row['label']] = weights[index]
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
def Entropy(Set, Labels, weights):
    totalEntropy = 0
    # debugFrac = 0
    for label in Labels:
        # print("Label: ", label, "Numerator: ", len(S.loc[S['label'] == label]), "Denominator: ", len(S))
        matchingLabel = Set.loc[Set['label'] == label]
        #print(len(matchingLabel))
        if len(matchingLabel) == 0:
            continue
        weightsList = [weights[i] for i in matchingLabel.index.values.tolist()]
        # print("Weights Indexes: ", matchingLabel.index.values.tolist())
        #print("All Weights:", weights)
        #print("Subset: ", weightsList)
        sumWeights = np.sum(weightsList)
        # print("SumWeights: ", sumWeights)
        frac = sumWeights# / len(S.index)
        # debugFrac += frac
        if frac == 0:
            continue
        elif frac >= 1 or frac < 0:
            print("Frac: ", frac)
        # print(-frac)
        # print(math.log(frac))
        # print("Sub-entropy: ", -frac * math.log(frac))
        totalEntropy += -frac * math.log(frac, 2)
        # totalEntropy += - len(S.loc[S['label'] == label]) / len(S) * math.log(len(S.loc[S['label'] == label]) / len(S))
    #print("Fraction total is: ", debugFrac)
    return totalEntropy

# returns the attribute that best splits the set S
def BestSplit(S, Attributes, Labels, chooser, weCareAboutUnknowns, weights):
    gains = {}
    initEntropy = Entropy(S, Labels, weights)
    #print("InitEntropy: ", initEntropy)
    allExamples = np.sum(weights)
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
                weightsList = [weights[i] for i in atrSubset.index.values.tolist()]
                atrScale = np.sum(weightsList) / allExamples
                # debugScale += atrScale
                # print("Scale: ", atrScale)
                entr = Entropy(atrSubset, Labels, weights)
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
            weightsList = [weights[i] for i in atrSubset.index.values.tolist()]
            atrScale = np.sum(weightsList) / allExamples
            # debugScale += atrScale
            entr = Entropy(atrSubset, Labels, weights)
            # print("-", (atrScale* entr))
            totalGains -= (atrScale * entr)
            # print("-Total: ", totalGains)

            atrSubset = S.loc[S[attribute] <= median]

            #print(attribute, ": ", atrSubset)
            weightsList = [weights[i] for i in atrSubset.index.values.tolist()]
            atrScale = np.sum(weightsList) / allExamples
            # debugScale += atrScale
            entr = Entropy(atrSubset, Labels, weights)
            # print("-", (atrScale* entr))
            totalGains -= (atrScale * entr)
            # print("-Total: ", totalGains)
        # print("Debug Scale: ", debugScale)
        gains[attribute] = totalGains
    greatestGains = None
    print("Gains:", gains)
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
def ID3(S, Attributes, Labels, depth, chooser, weCareAboutUnknowns, weights):
    allLabels, mostCommonLabel = GetTheDeets(S, weights)
    if len(list(allLabels.keys())) == 1:
        return Node({}, str(list(allLabels.keys())[0]), "")
    if len(Attributes) == 0 or depth < 1:
        return Node({}, mostCommonLabel, "")
    rootNode = Node({}, "", "")
    # print("Attributes pre BestSplit: ", Attributes)
    A = BestSplit(S, Attributes, allLabels.keys(), chooser, weCareAboutUnknowns, weights)
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
                rootNode.branches[value] = ID3(Sv, newAtr, Labels, depth - 1, chooser, weCareAboutUnknowns, weights)
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
            rootNode.branches[">"+str(median)] = ID3(Sv, newAtr, Labels, depth - 1, chooser, weCareAboutUnknowns, weights)
        Sv = S.loc[S[A] <= median]
        if len(Sv) == 0:
            rootNode.branches["<"+str(median)] = Node({}, mostCommonLabel, "")
        else:
            newAtr = Attributes.copy()
            newAtr.pop(A)
            rootNode.branches["<"+str(median)] = ID3(Sv, newAtr, Labels, depth - 1, chooser, weCareAboutUnknowns, weights)
    return rootNode

def TestTree(tree, testData, chooser):
    prediction = []
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
                elif branch[0] == '<':
                    if atrValue <= int(branch[1:-2]):
                        currentNode = currentNode.branches[branch]
                else:
                    # print(currentNode.branches)
                    # print(atrValue)
                    currentNode = currentNode.branches[atrValue]
                    break
        prediction.append(currentNode.label)
    accurate =0
    index = 0
    # print("Prediction: ", prediction)
    for label in prediction:
        if label == testData.iloc[index]['label']:
            accurate += 1
        index += 1
    print(chooser, ": ", accurate/index, "% accurate")

# returns a set mapping each data point to a prediction
def GetPredictions(S, InfoGainStump):
    prediction = [0] * len(S.index)
    for index, row in S.iterrows():
        currentNode = InfoGainStump
        while currentNode.attribute != "": # While this is not a leaf node
            atrValue = row[currentNode.attribute]
            # print(currentNode.attribute)
            # print(currentNode.branches)
            for branch in currentNode.branches.keys():
                # print("Branch: ", branch)
                # print("AtrValue: ", atrValue)
                if branch[0] == '>' and atrValue > int(branch[1:-2]):
                    currentNode = currentNode.branches[branch]
                    break
                elif branch[0] == '<' and atrValue <= int(branch[1:-2]):
                    currentNode = currentNode.branches[branch]
                    break
                elif atrValue == branch:
                    # print(currentNode.branches)
                    # print(atrValue)
                    currentNode = currentNode.branches[branch]
                    break
        prediction[index] = int(currentNode.label)
    # print("Prediction: ", prediction)
    return prediction

def GetError(predictions, weights, S):
    weightedError = 0
    i = 0
    #print("Predictions: ", predictions, "\nTrue: ", S['label'].tolist())
    for label in predictions:
        #print("Label: ", label, " True: ", S.loc[i].values.tolist()[-1])
        if label != S.loc[i].values.tolist()[-1]:
            weightedError += weights[i]
            # print("Weighted Error: ", weightedError)
        i += 1
    # print("Weighted Error: ", weightedError)
    # print(weights)
    return weightedError

def AdaBoost(S, Attributes, Labels, T, _D=None, _alpha=None, _h=None):
    if T == 1:
        D_t = [[0 for z in range(len(S.index))] for k in range(0, 501)]
        alpha_t = [0 for k in range(501)]
        D_t[0] = [1/len(S.index) for i in range(len(S.index))]
        hypotheses = [[] for k in range(501)]
        for t in range(0, T):
            # print("Before: ", np.sum(D_t[t-1]))
            InfoGainTree = ID3(S, Attributes, Labels, 1, "Information Gain", False, D_t[t])
            hypotheses[t] = GetPredictions(S, InfoGainTree)
            #print("Hypothesis: ", hypotheses[t])
            #print("Acutal: ", S['label'].tolist())
            error = GetError(hypotheses[t], D_t[t], S)
            if error == 0:
                alpha_t[t] = 0
            else:
                if error > 0.5:
                    print("Error: ", error)
                alpha_t[t] = 0.5 * np.log((1-error)/error)
            # z_t = np.sum(D_t[t-1])
            z_t = 0
            for i in range(len(S.index)):
                #print("I: ", i)
                #print("Predictions: ", hypotheses[t][i])
                #print("Actual: ", S.at[i, 'label'])
                isMatch = S.at[i, 'label'] * int(hypotheses[t][i])
                #print("isMatch: ", isMatch)
                #print("D_t_i: ", D_t[t][i])
                #print("alpha_t: ", alpha_t[t], "isMatch: ", isMatch)
                exponent = math.exp(-alpha_t[t]*isMatch)
                D_t[t+1][i] = D_t[t][i] * exponent
                z_t += D_t[t+1][i] + 0.000000001
            # print("Before: ", np.sum(D_t[t]))
            for j in range(0, len(D_t[t])):
                D_t[t+1][j] = D_t[t+1][j]/z_t
            # print("After: ", np.sum(D_t[t]))
        # errors[t] = GetError(predictions, D_t[t], S)
        #for line in hypotheses:
        #    print("Pre: ", line)
        finalHyp = [0 for k in range(len(S.index))]
        for x in range(len(S.index)):
            for t in range(T):
                finalHyp[x] += alpha_t[t]*int(hypotheses[t][x])
            finalHyp[x] = np.sign(finalHyp[x])
        return finalHyp, D_t, alpha_t, hypotheses
    else:
        D_t = _D
        alpha_t = _alpha
        hypotheses = _h
        t = T-1
        # print("Before: ", np.sum(D_t[t-1]))
        InfoGainTree = ID3(S, Attributes, Labels, 1, "Information Gain", False, D_t[t])
        print("Splitting on: ", InfoGainTree.attribute)
        hypotheses[t] = GetPredictions(S, InfoGainTree)
        #print("Hypothesis: ", hypotheses[t])
        #print("Acutal: ", S['label'].tolist())
        error = GetError(hypotheses[t], D_t[t], S)
        if error == 0:
            alpha_t[t] = 0
        else:
            if error > 0.5:
                print("Error: ", error)
            alpha_t[t] = 0.5 * np.log((1-error)/error)
        # z_t = np.sum(D_t[t-1])
        z_t = 0
        for i in range(len(S.index)):
            #print("I: ", i)
            #print("Predictions: ", hypotheses[t][i])
            #print("Actual: ", S.at[i, 'label'])
            isMatch = S.at[i, 'label'] * int(hypotheses[t][i])
            #print("isMatch: ", isMatch)
            #print("D_t_i: ", D_t[t][i])
            #print("alpha_t: ", alpha_t[t], "isMatch: ", isMatch)
            exponent = math.exp(-alpha_t[t]*isMatch)
            newWeight = D_t[t][i] * exponent
            D_t[t+1][i] = newWeight
            z_t += D_t[t+1][i] + 0.000000001
        # print("Before: ", np.sum(D_t[t]))
        for j in range(0, len(D_t[t])):
            D_t[t+1][j] = D_t[t+1][j]/z_t
        # print("After: ", np.sum(D_t[t]))
        # errors[t] = GetError(predictions, D_t[t], S)
        #for line in hypotheses:
        #    print("Pre: ", line)
        finalHyp = [0 for k in range(len(S.index))]
        for x in range(len(S.index)):
            for t in range(T):
                finalHyp[x] += alpha_t[t]*int(hypotheses[t][x])
            finalHyp[x] = np.sign(finalHyp[x])
        return finalHyp, D_t, alpha_t, hypotheses


def PlotErrorsOverT(finalHyp, weights, S, Test, T):
    x = [0 for k in range(0,T+1)]
    errors = [0 for k in range(0,T+1)]
    actual = S['label'].tolist()
    for i in range(1,T+1):
        x[i] = i
        error = 0
        #print('HYP:', finalHyp)
        #print('Actual: ', S['label'].tolist())
        for j in range(len(actual)):
            if finalHyp[i][j] != actual[j]:
                error += weights[i][j]
        errors[i] = math.exp(-2*pow(error, 2)*i)
    #print(arr)
    # print(errors)
    plt.plot(x, errors, 'g', label='training Data')
    actual = Test['label'].tolist()
    testError = [0 for k in range(0,T+1)]
    for k in range(1, T+1):
        error=0
        for l in range(len(actual)):
            if finalHyp[k][l] != actual[l]:
                error += weights[k][l]
        testError[k] = math.exp(-2*pow(error, 2)*k)
    plt.plot(x, testError, 'b', label='test Data')
    plt.legend()
    plt.xlabel("T Value")
    plt.ylabel("Error")
    plt.show()
    return 0

# Press the green button in the gutter to run the script.
def PlotWeightedErrors(allWeightedPredictions, weights, S, testData, T):
    x = [0 for z in range(0,T)]
    errors = [0 for z in range(0,T)]
    actual = S['label'].tolist()
    for i in range(0,T):
        x[i] = i
        error = 0
        #print('HYP:', finalHyp)
        #print('Actual: ', S['label'].tolist())
        for j in range(len(actual)):
            if allWeightedPredictions[i][j] != actual[j]:
                error += weights[i][j]
        errors[i] = error
    #print(arr)
    # print(errors)
    plt.plot(x, errors, 'g', label='training Data')
    actual = testData['label'].tolist()
    testError = [0 for k in range(0,T)]
    for k in range(T):
        error=0
        for l in range(len(actual)):
            if allWeightedPredictions[k][l] != actual[l]:
                error += weights[k][l]
        testError[k] = error
    plt.plot(x, testError, 'b', label='test Data')
    plt.legend()
    plt.xlabel("T Value")
    plt.ylabel("Error")
    plt.show()
    return 0


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

    depth = 1
    T = 500
    print("DEPTH=", depth)
    # print(initWeights)
    allHyp = [[] for k in range(0, T+1)]

    #S = S.head(30)
    #testData = testData.head(30)
    weights = None
    alpha = None
    hyp = None
    for t in range(1, T+1):
        print(t)
        allHyp[t], weights, alpha, hyp = AdaBoost(S, Attributes, Labels, t, _D=weights, _alpha=alpha, _h=hyp)
    toc = time.perf_counter()
    for line in allHyp:
        print(line)
    print("Actual: ", S['label'].tolist())
    print("Processing Time: ", str(toc-tic0))
    #PlotErrorsOverT(allHyp, weights, S, testData, T)
    PlotWeightedErrors(hyp, weights, S, testData, T)

