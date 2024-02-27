# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 08:47:00 2024

@author: HP
"""

"""
Entropy Information Gain and Gini Impurity

               Root Node
               
        
    Decision node                    Decision node
         or                               or
    Internal node                    Internal node
    
    
Terminal node
     or
leaf node
"""
"""
Entropy:
    Entropy is the quantittative measure of the randomness of the information
    Being proceed
    
    A high value of entropy is means that the randomnes in the system is high
    and thus making accurateprediction is tough
    
    A low value entropy means that the randomness in the system is low 
    and thus making accurate prediction is easier
    
Information Gain:
    is the measure of how much info. is feature provides about a class.
    Low entropy leads to increased information gain and high entropy leads
    to low info gain
    
    Information gain compute the diffrence between entropy before split 
    and average entropy after split of the database based on given feature
"""

"""
Gini impurity:
    The split made in a DEcision Tree is said to be to pure if all the 
    data ponts are accuretly separated into diffrent classes
    
    Gini impurity measures the likelihood that a randomly selected data 
    point would be incorrected classified by a specific mode
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing  import LabelEncoder
data=pd.read_csv("credit.csv")
data.isnull().sum()
data.dropna()
data.columns
data=data.drop(["phone"],axis=1)

#converting into binary
lb=LabelEncoder()
data["checking_balance"]=lb.fit_transform(data["checking_balance"])
data["credit_history"]=lb.fit_transform(data["credit_history"])
data["purpose"]=lb.fit_transform(data["purpose"])
data["savins_balance"]=lb.fit_transform(data["savins_balance"])
data["employement_duration"]=lb.fit_transform(data["employement_duration"])
data["other_credit"]=lb.fit_transform(data["other_credit"])
data["housing"]=lb.fit_transform(data["housing"])
data["job"]=lb.fit_transform(data["job"])
#################IMPORTANT#########################
data['default'].unique()
data["default"].value_counts()
colnames=list(data.columns)
#############################################################


predictors=colnames[:15]
target=colnames[15]

#spliting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test=train_test_split(data, test_size=0.3)
from sklearn.tree import DecisionTreeClassifier as DT

#help(DT)

model=DT(criterion='entropy')
model.fit(train[predictors],train[target])
preds=model.predict(test[predictors])
preds
pd.crosstab(test[target],preds,rownames=["Actual"],colnames=["predictions"])
np.mean(preds==test[target])

#now let us check accuracy on training dataset
preds_train=model.predict(train[predictors])
pd.crosstab(train[target],preds_train,rownames=['Actual'],colnames=["predictions"])
np.mean(preds_train==train[target])
#accuracy is less than train data and accuracy of greater than test data
#overfit the model

