# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Load the dataset Salary.csv using pandas and view the first few rows.
2.Check dataset information and identify any missing values.
3.Encode the categorical column "Position" into numerical values using LabelEncoder.
4.Define feature variables x as "Position" and "Level", and target variable y as "Salary".
5.Split the dataset into training and testing sets using an 80-20 split.
6.Create a DecisionTreeRegressor model instance.
7.Train the model using the training data.
8.Predict the salary values using the test data.
9.Evaluate the model using Mean Squared Error (MSE) and R² Score.
10.Use the trained model to predict salary for a new input [5, 6].
```
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: JEYAAARIKARAN P
RegisterNumber: 212224240064
*/
import pandas as pd
data=pd.read_csv('salary.csv')
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x = data[["Position", "Level"]]
y = data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.tree import DecisionTreeRegressor
data = DecisionTreeRegressor()
data.fit(x_train, y_train)
y_pred = data.predict(x_test)
from sklearn import metrics 
mse = metrics.mean_squared_error(y_test,y_pred)
mser2=metrics.r2_score(y_test,y_pred)
r2
data.predict([[5,6]])


```

## Output:
## Reading of dataset

![Screenshot 2025-04-28 211940](https://github.com/user-attachments/assets/573b23b1-f258-4cfe-965b-b3b914396a02)

## value of df.head()
![image](https://github.com/user-attachments/assets/81826ef2-6a73-4044-9430-c7577df1ee60)

## df.info()

![image](https://github.com/user-attachments/assets/b4181269-757e-4566-b6ab-220084330a5b)
## Value of df.isnull().sum()

![image](https://github.com/user-attachments/assets/c7588641-add2-4161-8f65-44896507c5f1)
## Data after encoding calculating Mean Squared Error

![image](https://github.com/user-attachments/assets/beb2688f-ea79-4955-97bf-914426e51b18)

## R2 value

![image](https://github.com/user-attachments/assets/1ac19062-158c-4bbe-8ad1-1369f692d6d5)
## Model prediction with [5,6] as input

![image](https://github.com/user-attachments/assets/7c9d6a84-c29a-4221-a941-03ed0f370472)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
