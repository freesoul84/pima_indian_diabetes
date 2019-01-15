import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split 

#dataset accessing
dataset=pd.read_csv("diabetes.csv")

#dataset decription and its groupying according to outcome
print("Dataset length:",len(dataset))
print(dataset.head())
print(dataset.describe())
print(dataset.groupby("Outcome").size())

# the scale of  feature is inconsistent
correlation=dataset.corr()
print("Correlation of datasets: ",correlation)
print("Correlation with outcome :",correlation['Outcome'].sort_values(ascending=False))
print("correlatin with age :",correlation['Age'].sort_values(ascending=False))

print("NUll values in dataset :",dataset.isnull().sum())

print((dataset['Glucose']==0).sum())
print((dataset['BloodPressure']==0).sum())
print((dataset['Insulin']==0).sum())
print((dataset['BMI']==0).sum())
print((dataset['SkinThickness']==0).sum())

#highly correlated feature scatterplot 
fig, a = plt.subplots()
a.scatter(dataset.iloc[:,1].values, dataset.iloc[:,2].values)
a.set_title('Highly Correlated Features')
a.set_xlabel('Glucose')
a.set_ylabel('Blood Pressure')
plt.show()

#correlation
heatmap= sns.heatmap(dataset)
print(heatmap)

#dataset splitting in test and train dataset
train,test = train_test_split(dataset, test_size=0.25, random_state=0, stratify=dataset['Outcome']) 
train_X = train[train.columns[:8]]
test_X = test[test.columns[:8]]
train_Y = train['Outcome']
test_Y = test['Outcome']

#model fitting
test_len=len(train_X)
neigh=int(np.sqrt(test_len))
model =  KNeighborsClassifier(n_neighbors=neigh)
model.fit(train_X,train_Y)

#prediction
prediction = model.predict(test_X)

print("Prediction:",prediction)

#confusion matrix creation
confusion_matrix = metrics.confusion_matrix(test_Y, prediction)
print("COnfusion matrix :",confusion_matrix)
print("accuracy score :",accuracy_score(test_Y, prediction))
print("f1 score :",f1_score(test_Y, prediction, average="macro"))
print("Precision_score :",precision_score(test_Y,prediction, average="macro"))
print("Prediction :",recall_score(test_Y, prediction, average="macro")) 