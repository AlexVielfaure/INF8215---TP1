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

        #print(np.ones((X.shape[0],1)).shape)
        #print(X.shape)
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
        #X_bias = np.concatenate(np.ones((X.shape[0],1)), X, axis=1)
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


#%% 2. Data preprocessing
# SI ON CHOISI DE NE PAS TOUCHER AU DATASET, LE MENTIONNER
# SI ON CHOISIT D'OMMETTRE UNE VARIABLE, LE MENTIONNER
# 1ere LIGNE c'est le nom des colonnes pour la soumission sur Kagle

def generate_missing_value_table(df):
        value = df.isnull().sum()
        value_percentage = 100 * df.isnull().sum() / len(df)
        table = pd.concat([value, value_percentage], axis=1)
        
        # rename the original table
        renamed_table = table.rename(columns = {0 : "missing values", 1 : "% of missing values"})
        renamed_table = renamed_table[renamed_table.iloc[:,1] != 0].sort_values("% of missing values", ascending=False).round(1)
        
        return renamed_table
#test = generate_missing_value_table(Raw_data)
#print(generate_missing_value_table(Raw_data))

PATH = "C:/Users/hatim/Documents/GitHub/INF8215---TP1/TP3/" # changer le path avec votre path
X_train = pd.read_csv("train.csv")
X_test = pd.read_csv("test.csv")
#y_train = X_train['Income']
#%% Missing values

# Replacing all " ?" with NaN
X_train_NaN = X_train.replace(" ",np.nan)
X_train_NaN = X_train.replace(" ?",np.nan)
# Finding all missing values
missing_values_table = generate_missing_value_table(X_train_NaN)
# Affichage
#display(missing_values_table)

#%% Changer les nan en unknown_classe
X_train_NaN["Occupation"] = X_train_NaN["Occupation"].replace(np.nan," Unknown_occ")
X_train_NaN["Workclass"] = X_train_NaN["Workclass"].replace(np.nan," Unknown_wc")
X_train_NaN["Native country"] = X_train_NaN["Native country"].replace(np.nan," Unknown_nc")

#%% Classifying classes
pd.options.mode.chained_assignment = None
# On change l'income en valeurs binaires
income_dict = {
                " <=50K" : 0,
                " <=50K." : 0,
                " >50K" : 1,
                " >50K." : 1,
                }
X_train_NaN = X_train_NaN.replace({"Income": income_dict})

# On retire l'indexe et le "Final weight", car ils sont non pertinents
X_train_NaN.drop(["index","Final weight"], axis = 1, inplace = True)


# Age
t1 = pd.DataFrame(X_train_NaN.groupby(['Age','Income']).size().unstack())/pd.DataFrame(X_train_NaN.groupby(['Age','Income']).size().unstack().sum(axis=1))
t1[1] = 1-t1[0]
t1.plot(kind='bar',stacked=True,title = 'Graphe du salaire en fonction du niveau de l age')

X_train_NaN["Age"][X_train_NaN["Age"]<20] = 20
X_train_NaN["Age"][X_train_NaN["Age"]>65] = 65
X_train_NaN["Age"] = np.digitize(X_train_NaN["Age"],np.arange(20,65,5)) - 1

#Worclass
a=pd.get_dummies(X_train_NaN["Workclass"])
X_train_NaN = pd.concat([X_train_NaN,a],axis = 1).drop(["Workclass"],axis = 1)

# Éducation
X_train_NaN.groupby(['Education','Income']).size().unstack().plot(kind='bar',stacked=True,title = 'Graphe du salaire en fonction du niveau d''éducation')

education_dict = {
                " HS-grad" : 1,
                " Some-college" : 3,
                " Bachelors" : 4,
                " Masters" : 5,
                " Assoc-voc" : 2,
                " 11th" : 0,
                " Assoc-acdm" : 2,
                " 10th" : 0,
                " 7th-8th" : 0,
                " Prof-school" : 6,
                " 9th" : 0,
                " 12th" : 0,
                " Doctorate" : 7,
                " 5th-6th" : 0,
                " 1st-4th" : 0,
                " Preschool" : 0
                }
X_train_NaN['Education'] = X_train_NaN['Education'].map(education_dict)

# Marital-status
X_train_NaN["Marital-status"].value_counts()
X_train_NaN.groupby(['Marital-status','Income']).size().unstack().plot(kind='bar',stacked=True,title='Graphe du salaire en fonction du niveau du statut marital')
marital_dict = {
                " Married-civ-spouse" : 'Married',
                " Never-married" : 'Not Married',
                " Divorced" : 'Not Married',
                " Separated" : 'Not Married',
                " Widowed" : 'Not Married',
                " Married-spouse-absent" : 'Married',
                " Married-AF-spouse" : 'Married',
                }
X_train_NaN['Marital-status'] = X_train_NaN['Marital-status'].map(marital_dict)
a=pd.get_dummies(X_train_NaN['Marital-status'])
X_train_NaN = pd.concat([X_train_NaN,a],axis = 1).drop(['Marital-status'],axis = 1)


# Occupation
X_train_NaN.groupby(['Occupation','Income']).size().unstack().plot(kind='bar',stacked=True,title='Graphe du salaire en fonction du niveau de l''occupation')
a=pd.get_dummies(X_train_NaN['Occupation'])
X_train_NaN = pd.concat([X_train_NaN,a],axis = 1).drop(['Occupation'],axis = 1)


# Relationship
X_train_NaN.groupby(['Relationship','Income']).size().unstack().plot(kind='bar',stacked=True,title='Graphe du salaire en fonction de la relation')
a=pd.get_dummies(X_train_NaN['Relationship'])
X_train_NaN = pd.concat([X_train_NaN,a],axis = 1).drop(['Relationship'],axis = 1)


# Sex
X_train_NaN.groupby(['Sex','Income']).size().unstack().plot(kind='bar',stacked=True,title='Graphe du salaire en fonction du sexe')
education_dict = {
                " Male" : 1,
                " Female" : 0,
                }
X_train_NaN['Sex'] = X_train_NaN['Sex'].map(education_dict)


#Capital gain/loss
X_train_NaN['Capital-gain'][X_train_NaN['Capital-gain'] < 5100] = 0
X_train_NaN['Capital-gain'][X_train_NaN['Capital-gain'] >= 5100] = 1
X_train_NaN['Capital-loss'][X_train_NaN['Capital-loss'] > 0] = -1
X_train_NaN['Capital-loss'] += 1

#Hours
#t1 = pd.DataFrame(X_train_NaN.groupby(['Hours per week','Income']).size().unstack())/pd.DataFrame(X_train_NaN.groupby(['Hours per week','Income']).size().unstack().sum(axis=1))
#t1[1] = 1-t1[0]
#t1.plot(kind='bar',stacked=True,title = 'Graphe du salaire en fonction du niveau de heure')


X_train_NaN["Hours per week"][X_train_NaN["Hours per week"] < 35] = 0
X_train_NaN['Hours per week'][(X_train_NaN['Hours per week'] < 45) & (X_train_NaN['Hours per week'] >= 35)] = 1
X_train_NaN['Hours per week'][X_train_NaN['Hours per week'] >= 45] = 2

#Country
country_dict = {
 ' United-States':'N_america',
 ' Mexico':'S_america'     ,                 
 ' Philippines':'Asia',
 ' Germany':'Europe',
 ' Puerto-Rico':'S_america',
 ' Canada':'N_america',
 ' India':'Asia',
 ' Cuba':'S_america',
 ' El-Salvador':'S_america',
 ' China':'Asia',                
 ' South':'S_america',                  
 ' Italy':'Europe',                     
 ' England':'Europe',
 ' Jamaica':'S_america', 
 ' Dominican-Republic':'S_america',
 ' Guatemala':'S_america', 
 ' Japan'  : 'Asia',                            
 ' Columbia' : 'S_america',                         
 ' Vietnam' : 'Asia',
 ' Poland' : 'Europe',
 ' Portugal' : 'Europe',
 ' Haiti' : 'S_america',
 ' Taiwan' : 'Asia',
 ' Greece' : 'Europe',
 ' Iran' : 'Asia',
 ' Nicaragua' : 'S_america',
 ' Peru' : 'S_america',
 ' Ireland': 'Europe',
 ' Ecuador' : 'S_america',
 ' Thailand' : 'Asia',
 ' France' : 'Europe',
 ' Cambodia' : 'Asia',
 ' Hong': 'Asia',
 ' Hungary' : 'Europe',
 ' Trinadad&Tobago' : 'Africa',
 ' Yugoslavia' : 'Europe',
 ' Laos' : 'Asia',
 ' Scotland' : 'Europe',                         
 ' Outlying-US(Guam-USVI-etc)': 'S_america',
 ' Honduras' : 'S_america',
                }
X_train_NaN['Native country'] = X_train_NaN['Native country'].map(country_dict)
a=pd.get_dummies(X_train_NaN['Native country'])
X_train_NaN = pd.concat([X_train_NaN,a],axis = 1).drop(['Native country'],axis = 1)


#%% Normalisation 

X_train = X_train_NaN / X_train_NaN.max()

#%% PCA

#from sklearn.decomposition import PCA
#
#pca = PCA(n_components=20)
#pca.fit(X_train)
#print(pca.explained_variance_ratio_)
#
#X_train2 = pca.fit_transform(X_train)
#

#%% 
from sklearn.preprocessing import LabelEncoder
y_train = X_train_NaN['Income']
X_train = X_train.drop('Income',axis=1)
target_label = LabelEncoder()
y_train_label = target_label.fit_transform(y_train)
print(target_label.classes_)

#%% Cross_validation
from sklearn.model_selection import cross_validate
def compare(models,X_train,y_train,nb_runs,scoring):
    
    scores = []
    for i in models:
        scores.append(cross_validate(i, X_train, y_train, cv=nb_runs, scoring = scoring))
        
    return scores

#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

X_train_preprocess = np.array(X_train)

nb_run = 3

models = [
    SoftmaxClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
]

scoring = ['neg_log_loss', 'precision_macro','recall_macro','f1_macro']

score = compare(models,X_train_preprocess,y_train_label,nb_run,scoring[3])

Score_dataframe = np.zeros((2,2))
for i in range(len(score)):
    Score_dataframe[i,0] = score[1]['test_score'].mean()
    Score_dataframe[i,1] = score[1]['test_score'].std()

Score_dataframe = pd.DataFrame(Score_dataframe)
Score_dataframe.columns = ['Moyenne','STD']
Score_dataframe.index = ['SoftmaxClassifier','RandomForest','GradientBoosting']


#%% Single Model testing

X_train_preprocess = np.array(X_train)

model_test = RandomForestClassifier()

test = cross_validate(model_test, X_train_preprocess, y_train, cv=2, scoring = 'f1_macro')


#%% Kaggle 

# best_model_1 = 
# pred_test = pd.Series(best_model_1.transform(X_test_preprocess))
# pred_test.to_csv("test_prediction_1.csv",index = False)
#
# best_model_2 = 
# pred_test = pd.Series(best_model_2.transform(X_test_preprocess))
# pred_test.to_csv("test_prediction_2.csv",index = False)

