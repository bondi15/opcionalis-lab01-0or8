#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 16:58:34 2017

@author: bondi
"""

import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def X_index(pixels,sorvaltas,oszlopvaltas,size):
    X = []
    temp = (pixels/2)*(pixels/(2-1))+(pixels/2)
    X.append(temp)
    for i in range(0,size):
        temp = temp + sorvaltas
        X.append(temp)
        temp = temp + oszlopvaltas
        X.append(temp)
    X = [int(x) for x in X]
    X = np.array(X)
    return X


def correct_index (labels_value, goodvalue):
    indices = np.isin(labels_value, goodvalue)
    goodindexes = np.array(np.concatenate(np.where(indices)))
    return goodindexes

def toocomplicated (pred, correct):
    count = 0
    for q in pred:
        for w in correct:
            if q == w:
                count = count +1
    return count
                
def evaluation (pred_eight, pred_zero, correct_eight, correct_zero):
    TP= toocomplicated(pred_eight, correct_eight)
    TN= toocomplicated(pred_zero, correct_zero)
    FN= toocomplicated(pred_zero, correct_eight)
    FP= toocomplicated(pred_eight, correct_zero)
    
    accuracy = (TP + TN)/(len(correct_eight) + len(correct_zero))
    f1_score = 2*TP/(2*TP+FP+FN)
    
    return accuracy, f1_score, TP, TN, FN, FP
    
    
    
labels = np.array(mnist.test.labels)
labels_value = np.array(np.argmax(mnist.test.labels, 1))


goodvalues = [0, 8]
indices = np.isin(labels_value, goodvalues)
goodindexes = np.array(np.concatenate((np.where(indices))))

test_images = np.array(mnist.test.images)
test_labels = np.array(mnist.test.labels)

zero_eight_images = np.take(test_images, goodindexes, axis=0)
zero_eight_labels = np.take(test_labels, goodindexes, axis=0)
zero_eight_labels = np.array(np.argmax(zero_eight_labels, 1))


pixels = 28
size = 6
sorvaltas = [28, -28]
oszlopvaltas = [1, -1]
X_index_list = []

for q in sorvaltas:
    for w in oszlopvaltas:
        X_index_list.append(X_index(pixels,q,w,size))

X_index_list = np.array(X_index_list)
X_index_list = np.concatenate(X_index_list)


W=[]
for x in range(0,pixels*pixels):
    if x in X_index_list:
        W.append(5)
    else:
        W.append(0)
    

values = np.dot(zero_eight_images, W)


index_eight = correct_index(zero_eight_labels, 8)
index_zero = correct_index(zero_eight_labels, 0)


treshold = 100
eight_pred = np.squeeze(np.array(np.where(values > treshold)))
zero_pred = np.squeeze(np.array(np.where(values < treshold)))


ACC, F1, TP, TN, FN, FP = evaluation(eight_pred, zero_pred, index_eight, index_zero)
print("Accuracy: ", ACC)
print("F1_score: ", F1)
print("pred 8 (correct 8): ", TP)   
print("pred 0 (correct 8): ", FN)  
print("pred 0 (correct 0): ", TN)  
print("pred 8 (correct 0): ", FP)  




        
