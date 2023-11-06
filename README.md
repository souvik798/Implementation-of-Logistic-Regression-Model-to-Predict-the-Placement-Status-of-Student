# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required libraries .
2.Load the dataset and check for null data values and duplicate data values in the dataframe.
3.Import label encoder from sklearn.preprocessing to encode the dataset.
4.Apply Logistic Regression on to the model.
5.Predict the y values.
6.Calculate the Accuracy,Confusion and Classsification report.

```
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SOUVIK KUNDU 
RegisterNumber: 212221230105

import pandas as pd
data=pd.read_csv('Placement_Data.csv') 
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state =0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## Output:
# 1.Placement Data
![1s](https://github.com/souvik798/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94752764/ae1557c5-46fd-421e-8eab-d55ea669b512)

# 2.salary Data
![2s](https://github.com/souvik798/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94752764/67617c20-76d1-4e5d-95cb-57705b1fcab1)


# 3.checking the null() function
![3s](https://github.com/souvik798/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94752764/ebd19429-bae7-4c47-be13-025ba87952d2)


# 4.Data Duplicate
![4s](https://github.com/souvik798/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94752764/bb855d45-6ace-4324-ac2e-0b2c0141d31d)


# 5.Print Data

![5s](https://github.com/souvik798/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94752764/df763da2-782e-4973-86d0-2e7dc042e7ed)

# 6.Data - status

![6s](https://github.com/souvik798/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94752764/b9d0103e-e28b-4c57-8f71-b03fda61fe37)

# 7. y_prediction array
![7s](https://github.com/souvik798/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94752764/a2264d00-f897-4cca-b7ea-43ff42b0b7fc)


# 8.Accuracy  value
![8s](https://github.com/souvik798/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94752764/a93b4fa7-ee15-4610-b1c5-5970c25df767)


# 9.Confusion array

![9s](https://github.com/souvik798/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94752764/63c3b8a2-ee14-4258-9ba0-cbba974822c0)


# 10.classification report

![10s](https://github.com/souvik798/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94752764/96fa482a-1608-44da-86bd-b049cc11f332)


# 11. Prediction for LR
![11s](https://github.com/souvik798/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94752764/75796b18-4f1e-4e51-80cf-5d25e96e1491)
rediction of LR


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
