# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### 1.Import the standard Libraries. 
#### 2.Set variables for assigning dataset values. 
#### 3.Import linear regression from sklearn. 
#### 4.Assign the points for representing in the graph. 
#### 5.Predict the regression for marks by using the representation of the graph. 
#### 6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```python
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Gautham Krishna S
RegisterNumber:  212223240036
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
### Dataset
![dataset](https://github.com/gauthamkrishna7/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/141175025/f97d31de-7938-4da3-b73f-5c2416212eea)

### Head Values
![head](https://github.com/gauthamkrishna7/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/141175025/893832a2-f4b7-4f08-8502-344f46972bbc)

### Tail Values
![tail](https://github.com/gauthamkrishna7/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/141175025/5814fbf9-d369-49e5-8185-ea7f2184ebcc)

### X and Y Values
![xyvalues](https://github.com/gauthamkrishna7/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/141175025/b9f2e4c1-4c92-42a3-a1ef-c8039afe6453)

### Predication values of X and Y
![predict ](https://github.com/gauthamkrishna7/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/141175025/d0dc14da-521e-48c1-9a08-7c7e5902b990)

### MSE,MAE and RMSE
![values](https://github.com/gauthamkrishna7/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/141175025/acca76e2-a712-4bd3-aa67-776ff3c19e4a)

### Training set
![train](https://github.com/gauthamkrishna7/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/141175025/d4e0dbe1-65e9-4343-8d15-1d601d027eb1)

### Testing Set
![test](https://github.com/gauthamkrishna7/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/141175025/90f0de95-b285-41a8-9b97-9bb09b4b2477)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
