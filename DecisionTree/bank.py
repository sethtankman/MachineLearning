import math
import pandas as pd

class Node:
    def __init__(self, branches, label, atr):
        self.branches = branches
        self.label = label
        self.attribute = atr

    def printNodes(self, finalString, numLevels):
        finalString += str(self.label) + "\n"
        if len(self.branches.keys()) == 0:
            return finalString
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

def BestSplit(S, Attributes, Labels, chooser):
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
        totalGains = initEntropy
        for value in Attributes[attribute]:
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
def ID3(S, Attributes, Labels, depth, chooser):
    allLabels, mostCommonLabel = GetTheDeets(S)
    if len(list(allLabels.keys())) == 1:
        return Node({}, str(list(allLabels.keys())[0]), "")
    if len(Attributes) == 0 or depth < 1:
        return Node({}, mostCommonLabel, "")
    rootNode = Node({}, "", "")
    # print("Attributes pre BestSplit: ", Attributes)
    A = BestSplit(S, Attributes, Labels, chooser)
    rootNode.attribute = A
    for value in Attributes[A]:
        # Adding a branch is done implicitly
        Sv = Subset(S, A, value)
        if len(Sv) == 0:
            rootNode.branches[value] = Node({}, mostCommonLabel, "")
        else:
            newAtr = Attributes.copy()
            # print("Attributes: ", Attributes)
            # print("Removing: ", A)
            newAtr.pop(A)
            rootNode.branches[value] = ID3(Sv, newAtr, Labels, depth - 1, chooser)
    return rootNode

def TestTree(tree, testData, chooser):
    prediction = []
    for index, row in testData.iterrows():
        currentNode = tree
        while currentNode.attribute != "":
            atrValue = row[currentNode.attribute]
            currentNode = currentNode.branches[atrValue]
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
    labelText = ""
    attrText = ""
    colText = ""
    with open("bank/data-desc.txt", 'r') as f:
        res = []
        line = f.readline()
        while line:
            if line[0] == '|':
                line = f.readline()
                total = ""
                while line and line[0] != "|":
                    total = total + line
                    line = f.readline()
                res.append((''.join(total.split())).split(','))
            else:
                line = f.readline()
    print("Res: ", res)
    Labels = res[0]
    print(Labels)
    Attributes = {}
    currentKey = ""
    for item in res[1]:
        if '.' in item:
            triplet = item.split('.')
            Attributes[currentKey].add(triplet[0])
            if triplet[1] != '':
                item = triplet[1]
            else:
                continue
        if ':' in item:
            (currentKey, firstValue) = item.split(':')
            Attributes[currentKey] = {firstValue}
        else:
            Attributes[currentKey].add(item)
    # print(Attributes)
    prefixes = list(Attributes.keys())
    prefixes.append('label')
    # print('Prefixes:', prefixes)
    S = pd.read_csv("bank/train.csv", header=0, names=prefixes)
    testData = pd.read_csv("bank/test.csv", header=0, names=prefixes)
    # print(S)
    # chooser can be Information Gain, Majority Error, or GINI
    # chooser="Information Gain"
    depth = 6
    InfoGainTree = ID3(S, Attributes, Labels, depth, "Information Gain")
    METree = ID3(S, Attributes, Labels, depth, "Majority Error")
    GINITree = ID3(S, Attributes, Labels, depth, "GINI")
    FakeTree = ID3(S, Attributes, Labels, depth, "Fake")
    # print(tree.printNodes("", 0))
    TestTree(InfoGainTree, testData, "Information Gain")
    TestTree(METree, testData, "Majority Error")
    TestTree(GINITree, testData, "GINI")
    TestTree(FakeTree, testData, "Fake")