# Multiple-Regression-Project
USING MACHINE LEARNING


----------- I am predicting the The PRICE OR SALES A THE AREA OF 5500, 2 BEDROOM, 2 BATHROOM, 2 STORIES AND 1 PARKING SPACE APARTMENT WOULD BE IN THE AREA -------

-- Understand the Dataset & cleanup (if required).
-- Build Regression models to predict the sales w.r.t a single & multiple feature
-- Also evaluate the model & it scores using R2.

import pandas as pd
from sklearn import linear_model
df = pd.read_csv("Housing.csv")
df
import pandas as pd
from sklearn import linear_model
import numpy as np

## defining the y variable based on independencies.
X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']]

# create a dictionary to map 'yes' and 'no' to 1 and 0
binary_map = {'yes': 1, 'no': 0}

# convert 'yes' to 1 and 'no' to 0 for the columns
X['mainroad'] = X['mainroad'].map(binary_map)
X['guestroom'] = X['guestroom'].map(binary_map)
X['basement'] = X['basement'].map(binary_map)
X['hotwaterheating'] = X['hotwaterheating'].map(binary_map)
X['airconditioning'] = X['airconditioning'].map(binary_map)
X['prefarea'] = X['prefarea'].map(binary_map)

# dependencies or the target
y = df['price']

## defining the function used
regr = linear_model.LinearRegression()
regr.fit(X, y)

## inserting the call out into prdeictedprice as the targeted out to predict on.
predictedprice = regr.predict([[5500, 2, 2, 2, 1, 1, 1, 0, 1, 0, 0]])
print(predictedprice)

RESULT: 6,245,060


-- To evaluate the model using R-squared, you need to first split the data into training and testing sets, fit the model on the training set, and then -- use it to make predictions on the test set. You can then calculate the R-squared value for the predictions and compare it to the R-squared value for a -- simple baseline model. Here's an example of how you can do this:

import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# load data
df = pd.read_csv("Housing.csv")

# select features and target variable
X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']]
y = df['price']

# create a dictionary to map 'yes' and 'no' to 1 and 0
binary_map = {'yes': 1, 'no': 0}

# convert 'yes' to 1 and 'no' to 0 for the columns
X['mainroad'] = X['mainroad'].map(binary_map)
X['guestroom'] = X['guestroom'].map(binary_map)
X['basement'] = X['basement'].map(binary_map)
X['hotwaterheating'] = X['hotwaterheating'].map(binary_map)
X['airconditioning'] = X['airconditioning'].map(binary_map)
X['prefarea'] = X['prefarea'].map(binary_map)

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# fit linear regression model on training data
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

# make predictions on test data
y_pred = regr.predict(X_test)

# calculate R-squared value for predictions and baseline model
r2 = r2_score(y_test, y_pred)
r2_baseline = r2_score(y_test, [y_train.mean()] * len(y_test))

print('R-squared for model: {:.3f}'.format(r2))
print('R-squared for baseline: {:.3f}'.format(r2_baseline))

RESULT: 
R-squared for model: 0.644
R-squared for baseline: -0.018
