# MULTILINEAR REGRESSION

# IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMPORTING DATASET
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

print("FIRST X ___\n",X)
print("FIRST y ___\n",y)

# Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

onehotencoder = OneHotEncoder(categorical_features= [3])
X = onehotencoder.fit_transform(X).toarray()

# Removing Dummy Variable Trap
X = X[:, 1:]

# Splitting Data into Training & Testing
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

print("POST-SCALED VALUES")
print(X_train)
print(X_test)
# DO YOU NEED TO SCALE DUMMY VARIABLES?
# IT DEPENDS on your models
# DO YOU NEED TO SCALE Y Variables?
# NO, Cz IT is dependent variable
"""
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set
y_pred = regressor.predict(X_test)

# Building Optimal Model for Backward Elimination Model
import statsmodels.formula.api as sm

# Add unit column for constant value to sustain the in records
X = np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog= y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog= y, exog=X_opt).fit()
regressor_OLS.summary()

y_pred1 = regressor.predict(X)
