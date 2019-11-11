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

        # self.nb_feature =
        # self.nb_classes = 

        # X_bias =         
        # self.theta_  = 
        

        for epoch in range( self.n_epochs):

            # logits = 
            # probabilities = 
            
            
            # loss =                
            # self.theta_ = 

            if self.early_stopping:
                pass


        return self


    def predict_proba(self, X, y=None):
        try:
            getattr(self, "theta_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        # X_bias = 
        pass

    
    def predict(self, X, y=None):
        try:
            getattr(self, "theta_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        pass

    
    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X,y)
 

    def score(self, X, y=None):
        pass
    
    
    def _cost_function(self,probabilities, y ): 
        pass
    

    def _one_hot(self,y):
        pass

    
    def _softmax(self,z):
        pass


    def _get_gradient(self,X_bias,y, probas):
        
        pass
    
#%%










