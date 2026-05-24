import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def train_linear(df, target):
    """
    Linear regression Training Loop
    """

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


def test_linear(df, target, model):
    """
    Testing Linear Regression Modles.
    """
    X =df.drop(columns=[target]).select_dtypes(include=['float64', 'int64']) #drop the target columns and only keep int64 and float64 dtypes

    y_pred = model.predict(X) #predict the targets using the features (X)

    return y_pred


def train_lasso(df, target):
    """
    Training Lasso Lasso Models
    """
    #-- Define features and target
    X = df.drop(columns=[target]).select_dtypes(include=['float64', 'int64'])
    y = df[target]

    #-- A Pipeline is used here to ensure the scaler is fit only on training folds, preventing data leakage during cross val.
    pipeline = Pipeline([
        ('scaler',StandardScaler()),
        ('lasso',LassoCV(cv=10))
    ])
    
    pipeline.fit(X,y)

    scores = cross_val_score(pipeline, X, y, cv=10, scoring='r2')
    y_pred_cv = cross_val_predict(pipeline, X, y, cv=10)
     
    return pipeline, scores, X, y, y_pred_cv

def test_lasso(df, target, model):
    """
    Testing Lasso Modles.
    """
    X =df.drop(columns=[target]).select_dtypes(include=['float64', 'int64']) #drop the target columns and only keep int64 and float64 dtypes

    y_pred = model.predict(X) #predict the targets using the features (X)

    return y_pred
