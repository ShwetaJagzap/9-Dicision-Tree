# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:13:11 2024

@author: HP
"""
"""Digitization and technology adoption to streamline operations and 
improve efficiency. Enhance customer experiences through digital 
channels. Optimize data-driven decision-making for business growth. 
Ensure cybersecurity measures to protect sensitive information. 
Foster innovation in digital products and services."""
#Decision tree 
# high space & computational complexity 
# low response time
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data=pd.read_csv("HR_DT.csv")

data.isnull().sum()
data.dropna()
data.columns
data=data.drop(["phone"],axis=1)

#converting into binary
lb=LabelEncoder()
data["no of Years of Experience of employee"]=lb.fit_transform(data["no of Years of Experience of employee"])
data["monthly income of employee"]=lb.fit_transform(data[" monthly income of employee"])

data['default'].unique()
data['default'].value_counts()
colnames=list(data.columns)

predictors=colnames[:15]
target=colnames[15]

from sklearn.model_selection import train_test_split
train, test= train_test_split(data, test_size=0.3)

from sklearn.tree import DecisionTreeClassifier as DT

model=DT(criterion='entropy')
model.fit(train[predictors],train[target])
preds=model.predict(test[predictors])
preds
pd.crosstab(test[target],preds,rownames=['Actual'],colnames=['predictions'])
np.mean(preds==test[target])


##############################################################

# now let us check accuracy on training dataset

model=DT(criterion='entropy')
model.fit(train[predictors],train[target])
preds_test=model.predict(test[predictors])
preds_test
pd.crosstab(test[target],preds,rownames=['Actual'],colnames=['predictions'])
np.mean(preds_test==test[target])


preds_train = model.predict(train[predictors])
pd.crosstab(train[target],preds_train,rownames=['Actual'],colnames=['predictions'])
np.mean(preds_train==train[target])

# accuracy on training data is 100%
# accuracy on test data i 69%
# hence the model is overfitted model