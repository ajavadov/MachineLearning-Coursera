# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 11:46:28 2019

@author: Aydin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('ex1data1.txt',header=None)
    
pop=data.iloc[:,0]
profit=data.iloc[:,1]
m = len(profit) # number of training example
data.head()

plt.scatter(pop, profit)
plt.xlabel('population in 10k')
plt.ylabel('profit in 10k$')
plt.show

pop=pop[:,np.newaxis]
profit=profit[:,np.newaxis]
theta=np.zeros([2,1])

iternum=1500
alpha=0.01
ones=np.ones([m,1])
pop=np.hstack((ones,pop))

def costFunc(pop,profit, theta):
    summing_term=(np.dot(pop,theta)) - profit
    return 1/(2*m)*np.sum(np.power(summing_term,2))

J=costFunc(pop,profit,theta)
print(J)



def gradient_desc(theta, iternum, alpha, pop, profit):
    
    for _ in range(iternum):
        theta = theta - (alpha/m) * pop.T@((pop @ theta)-profit)
    return theta

def normalEq(theta, pop, profit):
    theta=np.linalg.inv(pop.T @ pop) @ pop.T @ profit 
    return theta
    
opt_weights2=normalEq(theta, pop, profit)
print(opt_weights2)
opt_weights=gradient_desc(theta, iternum, alpha, pop, profit)
print(opt_weights)
        
new_J=costFunc(pop, profit,opt_weights)

plt.scatter(pop[:,1],profit);
plt.xlabel('population')
plt.ylabel('profit')
plt.plot(pop[:,1], pop @ opt_weights2)
def predict_one(pop, opt_weights, which):
    x1=pop[which,:][:,np.newaxis]
    return opt_weights.T @ x1


prediction=predict_one(pop, opt_weights, 29)
print(prediction)






