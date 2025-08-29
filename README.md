# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset from a CSV file and separate the features and target variable, encoding any categorical variables as needed.
2. Scale the features using a standard scaler to normalize the data.
3. Initialize model parameters (theta) and add an intercept term to the feature set.
4. Train the linear regression model using gradient descent by iterating through a specified number of iterations to minimize the cost function.
5. Make predictions on new data by transforming it using the same scaling and encoding applied to the training data.
 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: RAHUL S 
RegisterNumber: 212224040259
*/
```  
```
/*
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
def linear_reg(x,y,learning_rate=0.01,iters=1000):
    x=np.hstack((np.ones((x.shape[0],1)),x))
    theta=np.zeros((x.shape[1],1))
    for i in range(iters):
        prediction=x.dot(theta)
        error=prediction-y.reshape(-1,1)
        gradiant=(1/x.shape[0])*x.T.dot(error)
        theta-=learning_rate*gradiant
    return theta
df=pd.read_csv("/content/50_Startups.csv")
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
ct=ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[3])],remainder="passthrough")
x=ct.fit_transform(x)
y=y.astype(float)
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
theta=linear_reg(x_scaled,y,iters=1000,learning_rate=0.01)
new_data = np.array([165349.2, 136897.8, 471784.1, 'New York']).reshape(1,-1)
new_data_scaled=scaler.transform(ct.transform(new_data))
new_predict=np.dot(np.append(1,new_data_scaled),theta)
print(new_predict)
df.head()
*/
```

## Output:
![alt text](image.png)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
