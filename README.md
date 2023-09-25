# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset. 
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn
7. Apply new unknown values

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SOUVIK KUNDU 
RegisterNumber: 212221230105

# Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Read The File
dataset=pd.read_csv('Placement_Data_Full_Class.csv')
dataset
dataset.head(10)
dataset.tail(10)
# Dropping the serial number and salary column
dataset=dataset.drop(['sl_no','ssc_p','workex','ssc_b'],axis=1)
dataset
dataset.shape
dataset.info()
dataset["gender"]=dataset["gender"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset.info()
dataset["gender"]=dataset["gender"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset.info()
dataset
# selecting the features and labels
x=dataset.iloc[:, :-1].values
y=dataset.iloc[: ,-1].values
y
# dividing the data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
dataset.head()
y_train.shape
x_train.shape
# Creating a Classifier using Sklearn
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(random_state=0,solver='lbfgs',max_iter=1000).fit(x_train,y_train)
# Printing the acc
clf=LogisticRegression()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)
# Predicting for random value
clf.predict([[1	,78.33,	1,	2,	77.48,	2,	86.5,	0,	66.28]])


*/
```

## Output:

# read csv file:
![1](https://github.com/souvik798/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94752764/3ac53963-a59a-4b58-b7c6-d71c394ac16e)

# to read ten data(head):
![2](https://github.com/souvik798/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94752764/64b27700-8267-4f83-8159-d96770f31a7a)

# to read last ten data(tail):
![3](https://github.com/souvik798/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94752764/29d60e3e-2f56-4d5c-8212-40cb6dba69f7)

# Dropping the serial number and salary column:
![4](https://github.com/souvik798/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94752764/039bec45-7b14-4d70-b8f2-a535f4a51546)

# Dataset:
![5](https://github.com/souvik798/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94752764/067e105b-da36-4fad-b2ce-677858ecc584)

# Dataset information:
![6](https://github.com/souvik798/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94752764/8bf30832-fa2b-4c61-be02-68b98159b2e8)

# Dataset after changing object into category:
![7](https://github.com/souvik798/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94752764/4820752c-462f-4dd4-aa67-fb9c87ee49cf)

# dataset after changing category into integer :
![8](https://github.com/souvik798/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94752764/4cd7d93b-8198-4f5a-bc1e-b1836921d735)

# Displaying the dtaset:
![9](https://github.com/souvik798/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94752764/0eaa3641-6b03-4027-9da1-dc170364c9a3)

# selecting the  features and table:
![10](https://github.com/souvik798/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94752764/ad011149-eadd-40ce-b4be-e69b6de69fab)

# Dividing the data into train and test:
![11](https://github.com/souvik798/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94752764/900fe613-4493-488d-92bc-2e19f9d11993)

# shape ofx_train and y_train:
![12](https://github.com/souvik798/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94752764/08698371-7b8b-46bf-8144-d86b64ca3484)

#Predicting the random value:![13](https://github.com/souvik798/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94752764/b64bbba5-dd8d-45c0-a8f7-095e8424de67)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
