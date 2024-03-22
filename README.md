# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student
## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
Import the required packages and print the present data.
Print the placement data and salary data.
Find the null and duplicate values.
Using logistic regression find the predicted values of accuracy , confusion matrices.
Display the results.
## Program:
```
#Developed by: kabilan T
#RegisterNumber: 212222230059

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
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
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
### Placement Data:
![image](https://github.com/Afsarjumail/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343395/43bf5574-366c-4aa8-8bfd-2480d17a20a5)


### Salary Data:
![image](https://github.com/Afsarjumail/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343395/4481c3dc-e014-4f48-89c2-a05e28f9449e)

### Checking the null() function:
![image](https://github.com/Afsarjumail/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343395/f7bb1d3a-5411-4362-91b4-7017fdd1481d)


### Data Duplicate:
![image](https://github.com/Afsarjumail/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343395/6ac841e7-213e-4bc3-87a4-ff0f4818f536)

### Print Data:
![image](https://github.com/Afsarjumail/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343395/90b83a36-7f3e-45bd-8eb6-4f60883f3079)

### Data-Status:
![image](https://github.com/Afsarjumail/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343395/6f57b4f8-1050-4e17-a634-b3e6e6f8a286)

### Y_prediction array:
![image](https://github.com/Afsarjumail/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343395/390c1349-5449-4d2f-b9ec-27ce117d1395)


### Accuracy value:
![image](https://github.com/Afsarjumail/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343395/e069c29d-699f-43e4-b302-1c2546e24a67)


### Confusion array:
![image](https://github.com/Afsarjumail/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343395/42bdc9e2-8e65-424d-a13d-914a10cb1e40)


### Classification Report:
![image](https://github.com/Afsarjumail/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343395/1cefb2d7-c140-48a1-ae8e-be95c75652ec)


### Prediction of LR:
![image](https://github.com/Afsarjumail/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343395/d09d023b-6312-411e-944d-c165ffa2dbb0)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
