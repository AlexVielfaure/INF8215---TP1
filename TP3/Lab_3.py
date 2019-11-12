#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:46:13 2019

@author: alexandre
"""

import numpy as np
import math
import copy
import pandas as pd
import random
from sklearn.base import BaseEstimator, ClassifierMixin

#%%
class SoftmaxClassifier(BaseEstimator, ClassifierMixin):  

    def __init__(self, lr = 0.1, alpha = 100, n_epochs = 1000, eps = 1.0e-5,threshold = 1.0e-10 , early_stopping = True):
       
        self.lr = lr 
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.eps = eps
        self.threshold = threshold
        self.early_stopping = early_stopping
        


    """
        In:
        X : l'ensemble d'exemple de taille nb_example x nb_features
        y : l'ensemble d'étiquette de taille nb_example x 1

        Principe:
        Initialiser la matrice de poids
        Ajouter une colonne de bias à X
        Pour chaque epoch
            calculer les probabilités
            calculer le log loss
            calculer le gradient
            mettre à jouer les poids
            sauvegarder le loss
            tester pour early stopping

        Out:
        self, in sklearn the fit method returns the object itself
    """

    def fit(self, X, y=None):
        
        prev_loss = np.inf
        self.losses_ = []

        self.nb_feature = X.shape[1]
        self.nb_classes = np.max(y) + 1

        print(np.ones((X.shape[0],1)).shape)
        print(X.shape)
        X_bias = np.concatenate([np.ones((X.shape[0],1)), X], axis=1)   
        
        self.theta_  = np.random.rand(self.nb_feature + 1, self.nb_classes)
        
#        z = np.dot(X_bias,self.theta_) # un vecteur de dimension K qui correspond aux logits associés à x pour chacune des classes

        for epoch in range( self.n_epochs):

            probabilities =  self.predict_proba(X_bias,y)
            
            loss = self._cost_function(probabilities, y)            
            self.theta_ = self.theta_ - (self.lr * self._get_gradient(X_bias,y, probabilities))

            if self.early_stopping:
                if np.abs(loss - prev_loss) < self.threshold:
                    break

            prev_loss = loss
            
        return self


    def predict_proba(self, X, y=None):
        try:
            getattr(self, "theta_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
#        X_bias = np.concatenate(np.ones((X.shape[0],1)), X, axis=1)
        z = np.dot(X,self.theta_)
        
        p = self._softmax(z)
        
        return p

    
    def predict(self, X, y=None):
        try:
            getattr(self, "theta_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        X_bias = np.concatenate([np.ones((X.shape[0],1)), X], axis=1)
        p = self.predict_proba(X_bias)
        
        best_classes = np.argmax(p, axis = 1)
        
        return best_classes

    
    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X,y)
 

    def score(self, X, y=None):
        probabilities = self.predict_proba(X, y)
        log_loss = self._cost_function(probabilities, y)
        return log_loss
    
    
    def _cost_function(self,probabilities, y): 
        probabilities = np.clip(probabilities,self.eps,1-self.eps)
        new_y = self._one_hot(y)
        cost = new_y * np.log(probabilities)
        cost = cost.sum(axis = 1)
        cost = cost.sum(axis = 0)
        cost = -1 * cost / probabilities.shape[0]
        return cost
    

    def _one_hot(self,y):
        y_ohe = np.zeros((y.shape[0],self.nb_classes))
        for i,j in enumerate(y):
            y_ohe[i,j] = 1
        return y_ohe

    
    def _softmax(self,z):
        
        p = np.exp(z).T                     # Transpose to sum on rows
        p = (p / p.sum(axis = 0)).T
        
        return p


    def _get_gradient(self,X_bias,y, probas):
        gradient = (X_bias.T @ (probas-self._one_hot(y))) / X_bias.shape[0]
        return gradient
    
#%% 1.1
def testOneHot():
    softmax = SoftmaxClassifier()
    softmax.nb_classes = 6 # les classes possibles sont donc 0-5

    y1= np.array([0,1,2,3,4,5])
    y1.shape = (6,1)
    print('Premier test')
    print(softmax._one_hot(y1))

    y2 = np.array([5,5,5,5])
    y2.shape = (4,1)
    print('\nDeuxième test')
    print(softmax._one_hot(y2))

    y3 = np.array([0,0,0,0])
    y3.shape = (4,1)
    print('\nTroisième test')
    print(softmax._one_hot(y3))


testOneHot()

#%% 1.9
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load dataset
data,target =load_iris().data,load_iris().target

# split data in train/test sets
X_train, X_test, y_train, y_test = train_test_split( data, target, test_size=0.33, random_state=42)

# standardize columns using normal distribution
# fit on X_train and not on X_test to avoid Data Leakage
s = StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.transform(X_test)
#%%
cl = SoftmaxClassifier()

# train on X_train and not on X_test to avoid overfitting
train_p = cl.fit_predict(X_train,y_train)
test_p = cl.predict(X_test)
#%%
from sklearn.metrics import precision_recall_fscore_support

# display precision, recall and f1-score on train/test set
print("train : "+ str(precision_recall_fscore_support(y_train, train_p,average = "macro")))
print("test : "+ str(precision_recall_fscore_support(y_test, test_p,average = "macro")))




