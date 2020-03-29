#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 06:21:33 2020

@author: Hanqiu Deng <hanqiu.deng@outlook.com>
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import datasets
from sklearn.utils import shuffle
from minepy import MINE
from scipy.stats import spearmanr


def load_breast_cancer_data():
    breast_cancer = datasets.load_breast_cancer()
    feature_names = breast_cancer["feature_names"]
    breast_cancer_data = shuffle(breast_cancer.data)
    return breast_cancer_data,feature_names

def Spearmanr_matrix(data):
    corr = spearmanr(data).correlation
    RT = pd.DataFrame(corr)
    return RT

def MIC_matrix(data):
    mine = MINE()
    data = np.array(data)
    n = len(data[0, :])
    result = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            mine.compute_score(data[:, i], data[:, j])
            result[i, j] = mine.mic()
            result[j, i] = mine.mic()
    RT = pd.DataFrame(result)
    return RT

def ShowHeatMap(data_mic,data_spearmanr,feature_names):
    colormap = plt.cm.RdBu
    ylabels = feature_names
    plt.figure(figsize=(30, 15))
    ax = plt.subplot(1,2,1)
    ax.set_title('Fig(1) : Breast Cancer Features MIC HeatMap')
    sns.heatmap(data_mic.astype(float),
                cmap=colormap,
                ax=ax,
                annot=True,
                yticklabels=ylabels,
                xticklabels=ylabels)
    ax = plt.subplot(1,2,2)
    ax.set_title('Fig(2) : Breast Cancer Features Spearmanr HeatMap')
    sns.heatmap(data_spearmanr.astype(float),
                cmap=colormap,
                ax=ax,
                annot=True,
                yticklabels=ylabels,
                xticklabels=ylabels)
    plt.show()

def main():    
    breast_cancer_boston,feature_names = load_breast_cancer_data()
    data_mic = MIC_matrix(breast_cancer_boston)
    data_spearmanr = Spearmanr_matrix(data_mic)
    ShowHeatMap(data_mic,data_spearmanr,feature_names) 
    
if __name__ == '__main__':
    main()
