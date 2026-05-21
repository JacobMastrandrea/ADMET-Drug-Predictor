import numpy as np
from sklearn.linear_model import LinearRegression
from src.preprocessing import preprocess
from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error



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

    #--cross validated prediction 
    y_pred_cv = cross_val_predict(fitted_model, X, y, cv=10)

    return fitted_model, scores, X, y, y_pred_cv