# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 12:04:35 2019

@author: Aydin
"""

from scipy.io import loadmat
import numpy as np
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from PIL import Image
#let's read the data first.
data=loadmat('ex4data1.mat')
X=data['X']
Y=data['y']
_, axarr = plt.subplots(10,10,figsize=(9,9))
'''

'''
for i in range(10):
    for j in range(10):
       axarr[i,j].imshow(X[np.random.randint(X.shape[0])].reshape((20,20), order = 'F'))          
       axarr[i,j].axis('off')
weights = loadmat('ex4weights.mat')
theta1 = weights['Theta1']    #Theta1 has size 25 x 401
theta2 = weights['Theta2']    #Theta2 has size 10 x 26
#parametrlerin array daxili siralanmasi: 1ci layer, 1ci node-dan cixan weightler,2ci node-dan cixan 
#weightler......L-1 ci layerden axirinci nodedan cixan weightler
nn_params = np.hstack((theta1.ravel(order='F'), theta2.ravel(order='F')))    #unroll parameters
# neural network hyperparameters
input_layer_size = 400
hidden_layer_size = 25
num_labels = 10
lmbda = 1

def sigmoid(z):
    return 1/(1+np.exp(-z))

def nnCostFunc(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
    
    theta1 = np.reshape(nn_params[:hidden_layer_size*(input_layer_size+1)], (hidden_layer_size, input_layer_size+1), 'F')
    theta2 = np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):], (num_labels, hidden_layer_size+1), 'F')

    m = len(y)
    ones = np.ones((m,1))
    a1 = np.hstack((ones, X))
    a2 = sigmoid(a1 @ theta1.T)
    a2 = np.hstack((ones, a2))
    h = sigmoid(a2 @ theta2.T)
    
    y_d = pd.get_dummies(y.flatten())
    
    temp1 = np.multiply(y_d, np.log(h))
    temp2 = np.multiply(1-y_d, np.log(1-h))
    temp3 = np.sum(temp1 + temp2)
    
    sum1 = np.sum(np.sum(np.power(theta1[:,1:],2), axis = 1))
    sum2 = np.sum(np.sum(np.power(theta2[:,1:],2), axis = 1))
    
    return np.sum(temp3 / (-m)) + (sum1 + sum2) * lmbda / (2*m)


#nnCostFunc(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lmbda)

#sigmoid gradientinin funksiyasi
def sigmoidGradient(z):
    return np.multiply(sigmoid(z),1-sigmoid(z))

def random_initialize(LayerIn, LayerOut):
    epsilon=0.12
    return np.random.rand(LayerOut, LayerIn+1)*2*epsilon-epsilon

#initializing the weight matrices
    
theta1_initial=random_initialize(input_layer_size,hidden_layer_size)
theta2_initial=random_initialize(hidden_layer_size,num_labels)

neuralnets_initialparams=np.hstack((theta1_initial.ravel(order='F'),theta2_initial.ravel(order='F')))

def nnGrad(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
    
    theta1_initial = np.reshape(nn_params[:hidden_layer_size*(input_layer_size+1)], (hidden_layer_size, input_layer_size+1), 'F')
    theta2_initial = np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):], (num_labels, hidden_layer_size+1), 'F')
    y_d = pd.get_dummies(y.flatten())
    delta1 = np.zeros(theta1_initial.shape)
    delta2 = np.zeros(theta2_initial.shape)
    m = len(y)
    
    for i in range(X.shape[0]):
        ones = np.ones(1)
        a1 = np.hstack((ones, X[i]))
        z2 = a1 @ theta1_initial.T
        a2 = np.hstack((ones, sigmoid(z2)))
        z3 = a2 @ theta2_initial.T
        a3 = sigmoid(z3)

        d3 = a3 - y_d.iloc[i,:][np.newaxis,:]
        z2 = np.hstack((ones, z2))
        d2 = np.multiply(theta2_initial.T @ d3.T, sigmoidGradient(z2).T[:,np.newaxis])
        delta1 = delta1 + d2[1:,:] @ a1[np.newaxis,:]
        delta2 = delta2 + d3.T @ a2[np.newaxis,:]
    delta1 /= m
    delta2 /= m
    delta1[:,1:] = delta1[:,1:] + theta1_initial[:,1:] * lmbda / m
    delta2[:,1:] = delta2[:,1:] + theta2_initial[:,1:] * lmbda / m
        
    return np.hstack((delta1.ravel(order='F'), delta2.ravel(order='F')))

nnGrad(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lmbda)

#finding optimal weights using fmin_cg advanced optimization

theta_opt = opt.fmin_cg(maxiter = 50, f = nnCostFunc, x0 = neuralnets_initialparams, fprime = nnGrad, \
                        args = (input_layer_size, hidden_layer_size, num_labels, X, Y.flatten(), lmbda))

theta1_opt = np.reshape(theta_opt[:hidden_layer_size*(input_layer_size+1)], (hidden_layer_size, input_layer_size+1), 'F')
theta2_opt = np.reshape(theta_opt[hidden_layer_size*(input_layer_size+1):], (num_labels, hidden_layer_size+1), 'F')

#predict

def predict(theta1, theta2, X, y):
    m = len(y)
    ones = np.ones((m,1))
    a1 = np.hstack((ones, X))
    a2 = sigmoid(a1 @ theta1.T)
    a2 = np.hstack((ones, a2))
    h = sigmoid(a2 @ theta2.T)
    return np.argmax(h, axis = 1) +1

def predict_one(theta1, theta2, X):
    ons=np.ones((1,1))
    a1=np.hstack((ons,X))
    a2=sigmoid(a1 @ theta1.T)
    a2 = np.hstack((ons, a2))
    h=sigmoid(a2 @ theta2.T)
    return np.argmax(h,axis=1)+1
    


pred = predict(theta1_opt, theta2_opt, X, Y)
np.mean(pred == Y.flatten()) * 100



col = Image.open("six.png")
gray = col.convert('L')

# Let numpy do the heavy lifting for converting pixels to pure black or white
bw = np.asarray(gray).copy()
# Pixel range is 0...255, 256/2 = 128
bw[bw < 128] = 0  # Black
bw[bw >= 128] = 1 # White


for x in range(len(bw)):
    for y in range(len(bw)):
        if bw[x][y]==1:
            bw[x][y]=0
        else:
            bw[x][y]=1     
new_bw=bw.ravel(order="F")[np.newaxis,:]

five=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0.92, 0.86, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])




test=X[4906][np.newaxis,:]
netice=np.array([5])
newfive=five[np.newaxis,:]
pred=np.sum(predict_one(theta1_opt, theta2_opt, new_bw))
pred=0 if pred==10 else pred
print(pred)


newnetice=netice[np.newaxis, :]

np.mean(pred == newnetice.flatten()) * 100

