# Multiple-Regression-Project
USING MACHINE LEARNING


----------- I am predicting the The PRICE A THE AREA OF 5500, 2 BEDROOM, 2 BATHROOM, 2 STORIES AND 1 PARKING SPACE APARTMENT WOULD BE IN THE AREA -------

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
X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]
#dependencies or the target
y = df['price'] 

## defining the function used
regr = linear_model.LinearRegression()
regr.fit(X, y)

## inserting the call out into prdeictedprice as the targeted out to predict on.
predictedprice = regr.predict([[5500, 2, 2, 2, 1,]]) 

print(predictedprice)

RESULT: 5,751,976 

