# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 16:47:20 2024

@author: user
"""

'''Business Objective:
    Define the business objective, which could be predicting
    the likelihood of diabetes based on various features in 
    the dataset.
    
 Data Dictionary:
     
1) Pregnancies: Number of pregnancies (Integer)
2)Glucose: Plasma glucose concentration (integer)
3)BloodPressure: Diastolic blood pressure (mm Hg) (integer)
4)SkinThickness: Triceps skin fold thickness (mm) (integer)
5)Insulin: 2-Hour serum insulin (mu U/ml) (integer)
6)BMI: Body mass index (weight in kg / (height in m)^2) (float)
7)DiabetesPedigreeFunction: Diabetes pedigree function (float)
8)Age: Age of the patient (integer)
9)Outcome: Binary variable indicating whether the patient has diabetes or not (0: No, 1: Yes) (integer)'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder





from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#2.Load the Dataset
df=pd.read_csv("Fraud_check.csv")
df.head(5)
df.tail(5)
df.isnull().sum()
df.info()
df.columns
df.describe()


for i in df["Taxable.Income"]:
    if (i<=30000):
        df["Taxable.Income"]=np.where(df["Taxable.Income"]==i,0,df["Taxable.Income"])
    else:
        df["Taxable.Income"]=np.where(df["Taxable.Income"]==i,1,df["Taxable.Income"])
            
le=LabelEncoder()
df["Undergrad"]=le.fit_transform(df["Undergrad"])
df["Marital.Status"]=le.fit_transform(df["Marital.Status"])
df["Urban"]=le.fit_transform(df["Urban"])
df.head()

df["Taxable.Income"].value_counts()
inputs=df.drop(["Taxable.Income"],axis=1)
target=df["Taxable.Income"]

x_train,x_test,y_train,y_test=train_test_split(inputs,target,test_size=0.2)

model=DecisionTreeClassifier()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
y_pred

acc=accuracy_score(y_test, y_pred)
acc