# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:02:28 2019

@author: Aydin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

data=pd.read_csv('ex2data1.txt', header=None)

exams=data.iloc[:,0:2];
passed=data.iloc[:,2];
data.head()

 
# mask = passed == 1
# adm = plt.scatter(exams[mask][0].values, exams[mask][1].values)
# not_adm = plt.scatter(exams[~mask][0].values, exams[~mask][1].values)
# plt.xlabel('Exam 1 score')
# plt.ylabel('Exam 2 score')
# plt.legend((adm, not_adm), ('Admitted', 'Not admitted'))
# plt.show()
#

def mysigmo(f):
    return 1/(1+np.exp(-f))

def costFunc(exams, theta, passed):
    cost=(-1/m)*np.sum(np.multiply(passed,np.log(mysigmo(exams @ theta)))+np.multiply((1-passed),np.log(1-mysigmo(exams @ theta))))
    return cost
#def costFunction(theta, X, y):
#    J = (-1/m) * np.sum(np.multiply(y, np.log(sigmoid(X @ theta))) 
#        + np.multiply((1-y), np.log(1 - sigmoid(X @ theta))))
#    return J
#
#
#


m,n=exams.shape
mm=len(passed)
ones=np.ones((mm,1))
exams=np.hstack((ones, exams))
passed=passed[:,np.newaxis]
theta=np.zeros((3,1))

J=costFunc(exams, theta, passed)
print(J)
