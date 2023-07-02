# Original code from: https://github.com/m4jidRafiei/Decision-Tree-Python-
#
# Modified by a student to return the Digraph object instead of rendering it automatically.
# Modified to avoid error of mis-identification of graphviz nodes. Although I used a random
# generation and probabilistic cosmic rays might introduce equal IDs nevertheless.
import copy
import math
from collections import deque
from random import random

import numpy as np
from graphviz import Digraph


class Node(object):
    def __init__(self):
        self.value = None
        self.next = None
        self.childs = None
        self.name = ""

    def is_leaf(self):
        return self.childs is None

    def is_leaf_spec(self):
        return self.next is None


# Simple class of Decision Tree
# Aimed for who want to learn Decision Tree, so it is not optimized
class DecisionTree(object):
    def __init__(self, sample, attributes, labels, criterion):
        self.sample = sample
        self.attributes = attributes
        self.labels = labels
        self.labelCodes = None
        self.labelCodesCount = None
        self.initLabelCodes()
        self.criterion = criterion
        # print(self.labelCodes)
        self.gini = None
        self.entropy = None
        self.root = None
        if (self.criterion == "gini"):
            self.gini = self.getGini([x for x in range(len(self.labels))])
        else:
            self.entropy = self.getEntropy([x for x in range(len(self.labels))])

        self.num_nodes = []
        self.accuracy_train = []

    def initLabelCodes(self):
        self.labelCodes = []
        self.labelCodesCount = []
        for l in self.labels:
            if l not in self.labelCodes:
                self.labelCodes.append(l)
                self.labelCodesCount.append(0)
            self.labelCodesCount[self.labelCodes.index(l)] += 1

    def getLabelCodeId(self, sampleId):
        return self.labelCodes.index(self.labels[sampleId])

    def getAttributeValues(self, sampleIds, attributeId):
        vals = []
        for sid in sampleIds:
            val = self.sample[sid][attributeId]
            if val not in vals:
                vals.append(val)
        # print(vals)
        return vals

    def getEntropy(self, sampleIds):
        entropy = 0
        labelCount = [0] * len(self.labelCodes)
        for sid in sampleIds:
            labelCount[self.getLabelCodeId(sid)] += 1
        # print("-ge", labelCount)
        for lv in labelCount:
            # print(lv)
            if lv != 0:
                entropy += -lv / len(sampleIds) * math.log(lv / len(sampleIds), 2)
            else:
                entropy += 0
        return entropy

    def getGini(self, sampleIds):
        gini = 0
        labelCount = [0] * len(self.labelCodes)
        for sid in sampleIds:
            labelCount[self.getLabelCodeId(sid)] += 1
        # print("-ge", labelCount)
        for lv in labelCount:
            # print(lv)
            if lv != 0:
                gini += (lv / len(sampleIds)) ** 2
            else:
                gini += 0
        return 1 - gini

    def getDominantLabel(self, sampleIds):
        labelCodesCount = [0] * len(self.labelCodes)
        for sid in sampleIds:
            labelCodesCount[self.labelCodes.index(self.labels[sid])] += 1
        return self.labelCodes[labelCodesCount.index(max(labelCodesCount))]

    def getInformationGain(self, sampleIds, attributeId):
        gain = self.getEntropy(sampleIds)
        attributeVals = []
        attributeValsCount = []
        attributeValsIds = []
        for sid in sampleIds:
            val = self.sample[sid][attributeId]
            if val not in attributeVals:
                attributeVals.append(val)
                attributeValsCount.append(0)
                attributeValsIds.append([])
            vid = attributeVals.index(val)
            attributeValsCount[vid] += 1
            attributeValsIds[vid].append(sid)
        # print("-gig", self.attributes[attributeId])
        for vc, vids in zip(attributeValsCount, attributeValsIds):
            # print("-gig", vids)
            gain -= (vc / len(sampleIds)) * self.getEntropy(vids)
        return gain

    def getInformationGainGini(self, sampleIds, attributeId):
        gain = self.getGini(sampleIds)
        attributeVals = []
        attributeValsCount = []
        attributeValsIds = []
        for sid in sampleIds:
            val = self.sample[sid][attributeId]
            if val not in attributeVals:
                attributeVals.append(val)
                attributeValsCount.append(0)
                attributeValsIds.append([])
            vid = attributeVals.index(val)
            attributeValsCount[vid] += 1
            attributeValsIds[vid].append(sid)
        # print("-gig", self.attributes[attributeId])
        for vc, vids in zip(attributeValsCount, attributeValsIds):
            # print("-gig", vids)
            gain -= (vc / len(sampleIds)) * self.getGini(vids)
        return gain

    def getAttributeMaxInformationGain(self, sampleIds, attributeIds):
        attributesEntropy = [0] * len(attributeIds)
        for i, attId in zip(range(len(attributeIds)), attributeIds):
            attributesEntropy[i] = self.getInformationGain(sampleIds, attId)
        maxId = attributeIds[attributesEntropy.index(max(attributesEntropy))]
        try:
            maxvalue = attributesEntropy[maxId]
        except:
            maxvalue = 0
        return self.attributes[maxId], maxId, maxvalue

    def getAttributeMaxInformationGainGini(self, sampleIds, attributeIds):
        attributesEntropy = [0] * len(attributeIds)
        for i, attId in zip(range(len(attributeIds)), attributeIds):
            attributesEntropy[i] = self.getInformationGainGini(sampleIds, attId)
        maxId = attributeIds[attributesEntropy.index(max(attributesEntropy))]
        try:
            maxvalue = attributesEntropy[maxId]
        except:
            maxvalue = 0
        return self.attributes[maxId], maxId, maxvalue

    def isSingleLabeled(self, sampleIds):
        label = self.labels[sampleIds[0]]
        for sid in sampleIds:
            if self.labels[sid] != label:
                return False
        return True

    def getLabel(self, sampleId):
        return self.labels[sampleId]

    def id3(self, gain_threshold, minimum_samples):
        sampleIds = [x for x in range(len(self.sample))]
        attributeIds = [x for x in range(len(self.attributes))]
        self.root = self.id3Recv(sampleIds, attributeIds, self.root, gain_threshold, minimum_samples)

    def id3Recv(self, sampleIds, attributeIds, root, gain_threshold, minimum_samples):
        root = Node()  # Initialize current root
        if self.isSingleLabeled(sampleIds):
            root.value = self.labels[sampleIds[0]]
            return root
        # print(attributeIds)
        if len(attributeIds) == 0:
            root.value = self.getDominantLabel(sampleIds)
            return root
        if (self.criterion == "gini"):
            bestAttrName, bestAttrId, bestValue = self.getAttributeMaxInformationGainGini(sampleIds, attributeIds)
        else:
            bestAttrName, bestAttrId, bestValue = self.getAttributeMaxInformationGain(sampleIds, attributeIds)
        # print(bestAttrName)
        # if(bestValue > 0):
        # print("Best gain -> " + bestAttrName + "::" + str(bestValue) + "\n" )

        root.value = bestAttrName
        root.childs = []  # Create list of children

        if (bestValue < gain_threshold):
            Dominantlabel = self.getDominantLabel(sampleIds)
            root.value = Dominantlabel
            return root

        if (len(sampleIds) < minimum_samples):
            Dominantlabel = self.getDominantLabel(sampleIds)
            root.value = Dominantlabel
            return root

        for value in self.getAttributeValues(sampleIds, bestAttrId):
            # print(value)
            child = Node()
            child.value = value
            root.childs.append(child)  # Append new child node to current root
            childSampleIds = []
            for sid in sampleIds:
                if self.sample[sid][bestAttrId] == value:
                    childSampleIds.append(sid)
            if len(childSampleIds) == 0:
                child.next = self.getDominantLabel(sampleIds)
            else:
                # print(bestAttrName, bestAttrId)
                # print(attributeIds)
                if len(attributeIds) > 0 and bestAttrId in attributeIds:
                    toRemove = attributeIds.index(bestAttrId)
                    attributeIds.pop(toRemove)

                child.next = self.id3Recv(childSampleIds, attributeIds.copy(), child.next, gain_threshold,
                                          minimum_samples)
        return root

    def print_visualTree(self, render=True):
        dot = Digraph(comment='Decision Tree')
        if self.root:
            self.root.name = "root"
            roots = deque()
            roots.append(self.root)
            counter = 0
            while len(roots) > 0:
                root = roots.popleft()
                #                 print(root.value)
                dot.node(root.name, root.value)
                if root.childs:
                    for child in root.childs:
                        counter += 1
                        #                         print('({})'.format(child.value))
                        child.name = str(random())
                        dot.node(child.name, child.value)
                        dot.edge(root.name, child.name)
                        if (child.next.childs):
                            child.next.name = str(random())
                            dot.node(child.next.name, child.next.value)
                            dot.edge(child.name, child.next.name)
                            roots.append(child.next)
                        else:
                            child.next.name = str(random())
                            dot.node(child.next.name, child.next.value)
                            dot.edge(child.name, child.next.name)

                elif root.next:
                    dot.node(root.next, root.next)
                    dot.edge(root.value, root.next)
        #                     print(root.next)
        #         print(dot.source)
        if render:
            try:
                dot.render('output/visualTree.gv', view=True)
            except:
                print(
                    "You either have not installed the 'dot' to visualize the decision tree or the reulted .pdf file is open!")
        return dot

    def count_nodes(self, node):
        if not node.childs:
            return 1
        else:
            count = 1
            for child in node.childs:
                count += self.count_nodes(child.next)
            return count

    def get_num_nodes(self):
        return self.count_nodes(self.root)

    def traverse_tree_help(self, inp_data):

        return self.traverse_tree(self.root, inp_data)

    def traverse_tree(self, node, inp_data):
        dict = {'age': 0, 'workclass': 1, 'fnlwgt': 2, 'education': 3, 'education-num': 4, 'marital-status': 5,
                'occupation': 6, 'relationship': 7, 'race': 8, 'sex': 9, 'capital-gain': 10, 'capital-loss': 11,
                'hours-per-week': 12, 'native-country': 13}

        if node.value == ' <=50K' or node.value == ' >50k':
            return node.value

        if node.childs:
            for child in node.childs:
                data_index = dict[node.value]
                inp_value = inp_data[data_index]
                if child.value == inp_value:
                    return self.traverse_tree(child.next, inp_data)

    def reduced_error_pruning_helper(self, x_train_df, y_train_df):
        root = self.root

        for child in root.childs:
            self.reduced_error_pruning(child.next, x_train_df, y_train_df)

    def reduced_error_pruning(self, node, x_train_df, y_train_df):
        if node.value == " <=50K" or node.value == " >50k":
            return

        original_accuracy = self.evaluate_accuracy(x_train_df, y_train_df)
        # self.print_visualTree(render=True)
        # print("The node to be pruned: ", node.value)
        all_leafs = self.find_all_leafs(node)
        # find the majority value in the leafs
        np_array = np.array(all_leafs)
        num_0 = np.count_nonzero(np_array == " <=50K")
        num_1 = np.count_nonzero(np_array == " >50K")
        if num_0 > num_1:
            majority_class = " <=50K"
        else:
            majority_class = " >50K"

        print("The majority class: ", majority_class)

        # Delete the node
        cpy_node = copy.deepcopy(node)
        cpy_value = node.value
        cpy_childs = node.childs
        # the child is the majority class
        temp_node = Node()
        temp_node.value = majority_class
        temp_node.next = None
        temp_node.childs = None

        node.next = temp_node
        node.childs = None

        # self.print_visualTree(render=True)
        prune_accuracy = self.evaluate_accuracy(x_train_df, y_train_df)
        print("The accuracy before pruning: ", original_accuracy)
        print("The accuracy after pruning: ", prune_accuracy)
        # self.print_visualTree(render=True)
        num_nodes = self.get_num_nodes()
        print("Number of nodes: ", num_nodes)

        # push into array
        self.num_nodes.append(num_nodes)
        self.accuracy_train.append(prune_accuracy)

        if prune_accuracy <= original_accuracy:
            node.next = cpy_node
            node.value = cpy_value
            node.childs = cpy_childs
            return

        # node.value = majority_class
        node.next = majority_class

        self.reduced_error_pruning(node, x_train_df, y_train_df)

    def evaluate_accuracy(self, x_train_df, y_train_df):
        accuracy = 0

        # bar = Bar('Processing', max=len(x_train_df))
        for i in range(len(x_train_df)):
            dat_val = x_train_df.iloc[i]
            rec_val = self.traverse_tree_help(dat_val)
            if y_train_df.iloc[i] == rec_val:
                accuracy += 1
        #     bar.next()
        # bar.finish()
        return accuracy / len(x_train_df)

    def classify_data(self, node, data):
        if node.is_leaf():
            return node.value
        for child in node.childs:
            if child.name == data['attribute']:
                return self.classify_data(child, data)

    def find_all_leafs(self, node):
        if not node:
            return []
        if node.is_leaf_spec() and not node.childs:
            return [node.value]
        leafs = []
        # print("The parent node: ", node.value)
        if node.childs:
            for child in node.childs:
                # print("The child node: ", child.value)
                if child.next and child.next.value == '>50k' or child.next.value == '<=50k':
                    leafs.append(child.next.value)
                elif child.next and child.next.value != '>50k':
                    leafs += self.find_all_leafs(child.next)
                else:
                    leafs += self.find_all_leafs(child)
        return leafs
