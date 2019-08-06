# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:37:01 2019

@author: Aydin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data=pd.read_csv('ex1data2.txt',header=None)

features=data.iloc[:,0:2]
price=data.iloc[:,2]

m=len(price)


#feature normalization

features=(features-np.mean(features))/np.std(features)

ones=np.ones((m,1))
features=np.hstack((ones, features))
price=price[:, np.newaxis]

plt.scatter(features[:,1],price)
plt.xlabel('size')
plt.ylabel('price')

alpha = 0.01
iters = 4000
theta=np.zeros((3,1))

def costFunc(m,price, features, theta):
    cost=(1/(2*m))*np.sum(np.power(features@theta-price,2))
    return cost;
cost= costFunc(m,price, features, theta)
print(cost)

def normalEq(theta, features, price):
    theta=np.linalg.inv(features.T @ features) @ features.T @ price 
    return theta
opt_weights2=normalEq(theta, features, price)
print(opt_weights2)

def gradientDesc(theta, iters, alpha, features, price):
   
    for _ in range(iters):
        theta=theta-(alpha/m)*(features.T @(features@theta-price))
    return theta
new_theta=gradientDesc(theta, iters, alpha, features,price);


    
