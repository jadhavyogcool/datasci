# SIMPLE LINEAR REGRESSION

# IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMPORTING DATASET
dataset = pd.read_csv("1_Regression/Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# How does the current data look like?
plt.plot(X, y)

# Splitting Data into Training & Testing
# TODO Split in 80:20, i.e. 10 for test and 20 for training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# WELL DONE! YOUR DATA IS PREPROCESSED

# Fitting SIMPLE LINEAR REGRESSION to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

print('PREDICTED\tACTUAL')
for i in range(len(y_pred)):
    print(y_pred[i], '\t', y_train[i])

# Visualising the Training set results
# X axis = Employee Experience
# Y axis = Employee Salary
plt.scatter(X_train, y_train, color='red', label='Some1')
plt.plot(X_train, regressor.predict(X_train), color='blue', label='Some1')
plt.title('Salary vs. Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend(loc='upper left', numpoints=1)
plt.show()

# Visualising the Test set results
# X axis = Employee Experience
# Y axis = Employee Salary
plt.scatter(X_test, y_test, color='cyan')
plt.plot(X_train, regressor.predict(X_train), color='magenta')
plt.title('Salary vs. Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
