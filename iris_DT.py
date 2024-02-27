import pandas as pd
import numpy as np
from sklearn.preprocessing  import LabelEncoder
# Load the dataset
data = pd.read_csv('iris.csv')

# Display basic information about the dataset
print(data.info())

# Summary statistics
print(data.describe())

# Check for missing values
print(data.isnull().sum())
data=data.drop(["phone"],axis=1)

#converting into binary
lb=LabelEncoder()
data["sepal_length"]=lb.fit_transform(data["sepal_length"])
data["sepal_width"]=lb.fit_transform(data["sepal_width"])
data["petal_length"]=lb.fit_transform(data["petal_length"])
data["petal_width"]=lb.fit_transform(data["petal_width"])
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

