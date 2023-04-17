# Multiple-Regression-Project
USING MACHINE LEARNING


----------- I am predicting the The PRICE OR SALES A THE AREA OF 5500, 2 BEDROOM, 2 BATHROOM, 2 STORIES AND 1 PARKING SPACE APARTMENT WOULD BE IN THE AREA -------

-- Understand the Dataset & cleanup (if required).
-- Build Regression models to predict the sales w.r.t a single & multiple feature
-- Also evaluate the models & compare thier respective scores like R2.

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


-- The coefficient is a factor that describes the relationship with an unknown variable. 
-- Example: if x is a variable, then 5x is x five times. x is the unknown variable, and the number 5 is the coefficient.

import pandas
from sklearn import linear_model

X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]
y = df['price']

regr = linear_model.LinearRegression()
regr.fit(X, y)

print(regr.coef_)

RESULT: [3.31115495e+02 1.67809788e+05 1.13374016e+06 5.47939810e+05
 3.77596289e+05]
