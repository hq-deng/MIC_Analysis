#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 13:22:30 2020

@author: aesopc
"""

import numpy as np
import matplotlib.pyplot as plt
from minepy import MINE

def evaluate_without_noise(x,y,y1,y2):
    ax1 = plt.subplot(1,2,1)
    plt.sca(ax1)
    plt.plot(x,x)
    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.plot(x,y)
    result = calculate_mic(x,y)
    print(' ')
    print(' ')
    print(' ')
    print(' ')
    print('Calculate the MIC without noise:',result)

def evaluate_with_noise(x,y):
    ax1 = plt.subplot(1,2,2)
    plt.sca(ax1)
    np.random.seed(0)
    y += np.random.uniform(-5, 5, x.shape[0])
    plt.plot(x,y)
    result = calculate_mic(x,y)
    print(' ')
    print('Calculate the MIC with white noise:',result)
    print(' ')
    print(' ')
    print(' ')
    
    
def calculate_mic(x,y):
    mic = MINE()
    mic.compute_score(x, y)
    return mic.mic()
    
def main():
    x = np.linspace(0, 10, 1000)
    y1 = 10*np.sin(np.pi * x)
    y2 = (x-5)**2
    y = y2 + y1 + x   
    plt.figure(figsize = (16,8))
    evaluate_without_noise(x,y,y1,y2)
    evaluate_with_noise(x,y)
    plt.show()
    
if __name__ == '__main__':
    main()

