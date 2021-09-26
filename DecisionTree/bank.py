import math
import pandas as pd
import numpy as np

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
    for index, row in S.iterrows():
        if row[-1] in allLabels.keys():
            allLabels[row[-1]] += 1
        else:
            allLabels[row[-1]] = 1
    mostCommonLabel = None
    # print("All Labels:", allLabels)
    for eachLabel in allLabels:
        if mostCommonLabel is None or eachLabel[1] > mostCommonLabel[1]:
            mostCommonLabel = eachLabel
    return allLabels, mostCommonLabel


def Subset(S, A, value):
    return S.loc[S[A] == value]
    # return S where A = value

# divide number of examples with attributes by total number of attrubutes
# multiply that by -pos log(pos) - neg log(neg)
def Entropy(S, Labels):
    totalEntropy = 0
    for label in Labels:
        # print("Label: ", label, "Numerator: ", len(S.loc[S['label'] == label]), "Denominator: ", len(S))
        numerator = len(S.loc[S['label'] == label])
        if numerator == 0:
            continue
        frac = len(S.loc[S['label'] == label]) / len(S.index)
        totalEntropy += -frac * math.log(frac)
        # totalEntropy += - len(S.loc[S['label'] == label]) / len(S) * math.log(len(S.loc[S['label'] == label]) / len(S))
    return totalEntropy

def MajError(S, Labels):
    majError = -1
    for label in Labels:
        if len(S.index) == 0:
            return 1
        numWithLabel = len(S.loc[S['label'] == label]) / len(S.index)
        if numWithLabel > majError:
            majError = numWithLabel
    return 1 - majError

def GINI(S, Labels):
    gini = 1
    for label in Labels:
        if len(S.index) == 0:
            return 1
        frac = len(S.loc[S['label'] == label]) / len(S.index)
        gini -= frac * frac
    return gini

def BestSplit(S, Attributes, Labels, chooser, weCareAboutUnknowns):
    gains = {}
    initEntropy = -1
    if chooser == "Information Gain":
        initEntropy = Entropy(S, Labels)
    elif chooser == "Majority Error":
        initEntropy = MajError(S, Labels)
    elif chooser == "GINI":
        initEntropy = GINI(S, Labels)
    allAttributes = len(Attributes)
    for attribute in Attributes.keys():
        # print("Attribute: ", attribute)
        totalGains = initEntropy
        if Attributes[attribute][0] == 0:
            for value in Attributes[attribute][1]:
                if weCareAboutUnknowns and value == 'unknown':
                    otherVals = Attributes[attribute][1].copy()
                    #print("OTHER: ", otherVals)
                    otherVals.remove(value)
                    mostCommonAtrVal = ("", -1)
                    for contestVal in otherVals:
                        contestAtrNum = len(S.loc[S[attribute] == contestVal])
                        if contestAtrNum > mostCommonAtrVal[1]:
                            mostCommonAtrVal = (contestVal, contestAtrNum)
                    value = mostCommonAtrVal[0]
                atrSubset = Subset(S, attribute, value)
                atrScale = len(atrSubset) / allAttributes
                entr = -1
                if chooser == "Information Gain":
                    entr = Entropy(atrSubset, Labels)
                elif chooser == "Majority Error":
                    entr = MajError(atrSubset, Labels)
                elif chooser == "GINI":
                    entr = GINI(atrSubset, Labels)
                totalGains -= (atrScale * entr)
        else:
            numericValues = S[attribute].tolist()
            median = np.median(numericValues)
            #print("numeric: ", numericValues, "\nMedian: ", median)
            atrSubset = S.loc[S[attribute] > median]
            #print(attribute, ": ", atrSubset)
            atrScale = len(atrSubset) / len(S.index)
            entr = -1
            if chooser == "Information Gain":
                entr = Entropy(atrSubset, Labels)
            elif chooser == "Majority Error":
                entr = MajError(atrSubset, Labels)
            elif chooser == "GINI":
                entr = GINI(atrSubset, Labels)
            totalGains -= (atrScale * entr)

            atrSubset = S.loc[S[attribute] <= median]

            #print(attribute, ": ", atrSubset)
            atrScale = 1 - atrScale
            entr = -1
            if chooser == "Information Gain":
                entr = Entropy(atrSubset, Labels)
            elif chooser == "Majority Error":
                entr = MajError(atrSubset, Labels)
            elif chooser == "GINI":
                entr = GINI(atrSubset, Labels)
            totalGains -= (atrScale * entr)
        gains[attribute] = totalGains
    greatestGains = -1
    for atr in gains.keys():
        if greatestGains == -1 or gains[atr] > greatestGains[1]:
            greatestGains = (atr, gains[atr])
    return greatestGains[0]


# The ID3 algorithm
# S = The set of examples
# Attributes = the set of measured attributes
# Label = The target attribute (prediction)
def ID3(S, Attributes, Labels, depth, chooser, weCareAboutUnknowns):
    allLabels, mostCommonLabel = GetTheDeets(S)
    if len(list(allLabels.keys())) == 1:
        return Node({}, str(list(allLabels.keys())[0]), "")
    if len(Attributes) == 0 or depth < 1:
        return Node({}, mostCommonLabel, "")
    rootNode = Node({}, "", "")
    # print("Attributes pre BestSplit: ", Attributes)
    A = BestSplit(S, Attributes, Labels, chooser, weCareAboutUnknowns)
    rootNode.attribute = A
    if Attributes[A][0] == 0:
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
    else:
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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    Labels = ['yes', 'no']
    #print(Labels)
    # attrubute: (isNumeric, list)
    Attributes = {'age': (1, []), 'job': (0, ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
                                    "blue-collar","self-employed","retired","technician","services"]),
                  'marital': (0, ["married","divorced","single"]), 'education': (0, ["unknown","secondary","primary","tertiary"]),
                  'default': (0, ["yes","no"]), 'balance': (1, []), 'housing': (0, ["yes","no"]), 'loan': (0, ["yes","no"]),
                  'contact': (0, ["unknown","telephone","cellular"]), 'day': (1, []),
                  'month': (0, ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]),
                  'duration': (1, []), 'campaign': (1, []), 'pdays': (1, []), 'previous': (1, []),
                  'poutcome': (0, ["unknown","other","failure","success"])}
    # print(Attributes)
    prefixes = list(Attributes.keys())
    prefixes.append('label')
    # print('Prefixes:', prefixes)
    S = pd.read_csv("bank/train.csv", header=0, names=prefixes)
    testData = pd.read_csv("bank/test.csv", header=0, names=prefixes)
    # print(S)
    # chooser can be Information Gain, Majority Error, or GINI
    # chooser="Information Gain"
    depth = 1
    print("TEST DATA - We care about unknowns")
    while depth < 17:
        print("DEPTH=", depth)
        InfoGainTree = ID3(S, Attributes, Labels, depth, "Information Gain", True)
        METree = ID3(S, Attributes, Labels, depth, "Majority Error", True)
        GINITree = ID3(S, Attributes, Labels, depth, "GINI", True)
        # FakeTree = ID3(S, Attributes, Labels, depth, "Fake")
        #print(InfoGainTree.printNodes("", 0))
        TestTree(InfoGainTree, testData, "Information Gain")
        TestTree(METree, testData, "Majority Error")
        TestTree(GINITree, testData, "GINI")
        # TestTree(FakeTree, testData, "Fake")
        depth += 1

    depth = 1
    print("TRAINING DATA - We care about unknowns")
    while depth < 17:
        print("DEPTH=", depth)
        InfoGainTree = ID3(S, Attributes, Labels, depth, "Information Gain", True)
        METree = ID3(S, Attributes, Labels, depth, "Majority Error", True)
        GINITree = ID3(S, Attributes, Labels, depth, "GINI", True)
        # FakeTree = ID3(S, Attributes, Labels, depth, "Fake")
        # print(InfoGainTree.printNodes("", 0))
        TestTree(InfoGainTree, S, "Information Gain")
        TestTree(METree, S, "Majority Error")
        TestTree(GINITree, S, "GINI")
        # TestTree(FakeTree, testData, "Fake")
        depth += 1

    depth = 1
    print("TRAINING DATA - We DON'T care about unknowns")
    while depth < 17:
        print("DEPTH=", depth)
        InfoGainTree = ID3(S, Attributes, Labels, depth, "Information Gain", False)
        METree = ID3(S, Attributes, Labels, depth, "Majority Error", False)
        GINITree = ID3(S, Attributes, Labels, depth, "GINI", False)
        # FakeTree = ID3(S, Attributes, Labels, depth, "Fake")
        # print(InfoGainTree.printNodes("", 0))
        TestTree(InfoGainTree, S, "Information Gain")
        TestTree(METree, S, "Majority Error")
        TestTree(GINITree, S, "GINI")
        # TestTree(FakeTree, testData, "Fake")
        depth += 1

    depth = 1
    print("TEST DATA - We DON'T care about unknowns")
    while depth < 17:
        print("DEPTH=", depth)
        InfoGainTree = ID3(S, Attributes, Labels, depth, "Information Gain", False)
        METree = ID3(S, Attributes, Labels, depth, "Majority Error", False)
        GINITree = ID3(S, Attributes, Labels, depth, "GINI", False)
        # FakeTree = ID3(S, Attributes, Labels, depth, "Fake")
        # print(InfoGainTree.printNodes("", 0))
        TestTree(InfoGainTree, testData, "Information Gain")
        TestTree(METree, testData, "Majority Error")
        TestTree(GINITree, testData, "GINI")
        # TestTree(FakeTree, testData, "Fake")
        depth += 1