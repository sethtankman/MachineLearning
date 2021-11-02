import os
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
                if weCareAboutUnknowns and value == '?':
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

def TestTrees(trees, testData, chooser, T, prediction=None):
    if prediction == None:
        prediction = [0 for k in range(len(testData.index))]
    tree = trees[T]
    for index, row in testData.iterrows():
        currentNode = tree
        while currentNode.attribute != "":
            atrValue = row[currentNode.attribute]
            #print(currentNode.attribute)
            #print(atrValue + "\n")
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
                elif branch == atrValue or atrValue == '?':
                    # print(currentNode.branches)
                    # print(atrValue)
                    currentNode = currentNode.branches[branch]
                    break
        prediction[index] += int(currentNode.label)
    accurate = 0
    index = 0
    # print("Prediction: ", prediction)
    for label in prediction:
        finalPrediction = (label/(T+1)) >= 0.5
        if finalPrediction == testData.iloc[index]['label']:
            accurate += 1
        #else:
        #    print(index)
        index += 1
    print(chooser, ": ", accurate/index, "% accurate")
    return accurate/index, prediction

def PlotAccuracy(training, test, T):
    x = [0 for z in range(T)]
    for i in range(T):
        x[i] = i
    #print(arr)
    # print(errors)
    plt.plot(x, training, 'g', label='training Data')
    #plt.plot(x, test, 'b', label='test Data')
    plt.legend()
    plt.xlabel("T Value")
    plt.ylabel("% Accuracy")
    plt.show()
    return 0


def GetTestPrediction(testData, trees, T):
    prediction = [0 for k in range(len(testData.index))]
    for t in range(T):
        # print(t)
        tree = trees[t]
        for index, row in testData.iterrows():
            # print(index)
            currentNode = tree
            while currentNode.attribute != "":
                atrValue = row[currentNode.attribute]
                #print(currentNode.attribute)
                #print(atrValue + "\n")
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
                    elif branch == atrValue or atrValue == '?':
                        # print(currentNode.branches)
                        # print(atrValue)
                        currentNode = currentNode.branches[branch]
                        break
            prediction[index] += int(currentNode.label)
    prediction = np.array(prediction)
    return prediction/T


if __name__ == '__main__':
    tic0 = time.perf_counter()
    Labels = [0, 1]
    # attrubute: (isNumeric, list)
    Attributes = {'age': (1, []), 'workclass': (0, ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
                                                    'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']),
                  'fnlwgt': (1, []), 'education': (0, ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school',
                                                   'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters',
                                                   '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']),
                  'education-num': (1, []), 'marital-status': (0, ['Married-civ-spouse', 'Divorced', 'Never-married',
                                                                   'Separated', 'Widowed', 'Married-spouse-absent',
                                                                   'Married-AF-spouse']),
                  'occupation': (0, ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
                                     'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical',
                                     'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']),
                  'relationship': (0, ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']),
                  'race': (0, ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']),
                  'sex': (0, ['Female', 'Male']), 'capital-gain': (1, []), 'capital-loss': (1, []),
                  'hours-per-week': (1, []), 'native-country': (0, ['United-States', 'Cambodia', 'England',
                                                                    'Puerto-Rico', 'Canada', 'Germany',
                                                                    'Outlying-US(Guam-USVI-etc)', 'India', 'Japan',
                                                                    'Greece', 'South', 'China', 'Cuba', 'Iran',
                                                                    'Honduras', 'Philippines', 'Italy', 'Poland',
                                                                    'Jamaica', 'Vietnam', 'Mexico', 'Portugal',
                                                                    'Ireland', 'France', 'Dominican-Republic', 'Laos',
                                                                    'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary',
                                                                    'Guatemala', 'Nicaragua', 'Scotland', 'Thailand',
                                                                    'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago',
                                                                    'Peru', 'Hong', 'Holand-Netherlands'])}
    prefixes = list(Attributes.keys())
    testData = pd.read_csv("./test_final/test_final.csv", header=0)
    prefixes.append('label')
    S = pd.read_csv("./train_final/train_final.csv", header=0, names=prefixes)
    ids = testData['ID']
    del testData['ID']
    testData.set_axis(Attributes.keys(), axis=1)

    S = S.head(100)
    #testData = testData.head(100)

    depth = 5
    T = 50
    trainingAcc = [0 for k in range(T)]
    testAcc = [0 for k in range(T)]
    Classifier = []
    predictionTraining = None
    predictionTest = None
    for t in range(T):
        print(t)
        randomIndexes = random.choices(population=S.index.tolist(), k=len(S.index))
        mPrime = S.iloc[randomIndexes]
        Classifier.append(ID3(mPrime, Attributes, Labels, depth, "Information Gain", True))
        trainingAcc[t], predictionTraining = TestTrees(Classifier, S, "Information Gain", t, prediction=predictionTraining)
        #testAcc[t], predictionTest = TestTrees(Classifier, testData, "Information Gain", t, prediction=predictionTest)


    toc = time.perf_counter()
    print("Processing Time: ", str(toc-tic0))
    PlotAccuracy(trainingAcc, testAcc, T)

    #pred = GetTestPrediction(testData, Classifier, T)
    #print("XYZ")
    #df = pd.DataFrame({'Prediction':pred})
    #ids = pd.DataFrame(ids)
    #finalDF = ids.join(df)
    #print(os.getcwd())
    #finalDF.to_csv(os.getcwd() + '/smallOutputBag.csv', index=False)
    #finalDF.to_csv(os.getcwd() + '/testPredictionsBag.csv', index=False)
    #print("Oh hi mark")
