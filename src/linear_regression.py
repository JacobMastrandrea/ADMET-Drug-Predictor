import numpy as np
from sklearn.linear_model import LinearRegression
from src.preprocessing import preprocess
from sklearn.model_selection import cross_val_score



def train(df, target):

    #-- Define features and target
    X = df.drop(columns=[target]).select_dtypes(include=['float64', 'int64'])
    y = df[target]

    #-- Define instance of linear regression model
    lin_reg_model = LinearRegression()

    #-- fit model
    fitted_model = lin_reg_model.fit(X,y)

    #-- cross validation
    scores = cross_val_score(fitted_model, X, y, cv=10, scoring='r2')

    return fitted_model, scores, X, y