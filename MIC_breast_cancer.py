#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 06:11:14 2020

@author: Hanqiu Deng <hanqiu.deng@outlook.com>
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from minepy import MINE


def load_data():
    breast_cancer = datasets.load_breast_cancer()
    feature_names = breast_cancer["feature_names"]
    X, Y = shuffle(breast_cancer.data, breast_cancer.target, random_state=4)
    return X,Y,feature_names

def mic(X,Y):
    calculate_mic = MINE()
    result = []
    for i in range(X.shape[-1]):
        calculate_mic.compute_score(X[:,i], Y)
        result.append(calculate_mic.mic())
    return result

def importance_relative(X,Y):
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=43)
    X_train,Y_train = X,Y

    clf = RandomForestClassifier(n_estimators=500, random_state=4)
    clf.fit(X_train, Y_train)
    #print("Accuracy on test data: {:.2f}".format(clf.score(X_test, Y_test)))
    importance_result = clf.feature_importances_
    return importance_result

def visualization(mic_result,importance_result, feature_names):
    #MIC
    mic_result = np.array(mic_result)
    sorted_idx = np.argsort(mic_result)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1,2,1)
    plt.barh(pos, mic_result[sorted_idx], align='center')
    plt.yticks(pos, feature_names[sorted_idx])
    plt.xlabel('Fig(1) : Maximal Information Coefficient')
    plt.title('Correlation Degree')
    #Importance
    importance_result = 100.0 * (importance_result / importance_result.max())
    sorted_idx = np.argsort(importance_result)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, importance_result[sorted_idx], align='center')
    plt.yticks(pos, feature_names[sorted_idx])
    plt.xlabel('Fig(2) : Relative Importance')
    plt.title('Variable Importance')
    plt.show()
    
def main():    
    X,Y,feature_names = load_data()
    mic_result = mic(X,Y)
    plt.figure(figsize=(26, 13))
    importance_result = importance_relative(X,Y)
    visualization(mic_result,importance_result,feature_names)    
    
if __name__ == '__main__':
    main()