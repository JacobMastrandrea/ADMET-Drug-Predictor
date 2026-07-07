from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_val_predict


def train_random_forest(df, target):
    """
    A function to train a random forest model on the data passed through
    """
    all_targets = ['KSOL','LogD', 'log_HLM CLint', 'log_MLM CLint']
    #-- Define features and target
    X = df.drop(columns=all_targets).select_dtypes(include=['float64', 'int64'])
    y = df[target]

    #-- Define instance of random forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    #-- fit model
    fitted_model = rf_model.fit(X, y)

    #-- cross validation
    scores = cross_val_score(fitted_model, X, y, cv=10, scoring='r2')

    #-- cross validated prediction 
    y_pred_cv = cross_val_predict(fitted_model, X, y, cv=10)

    return fitted_model, scores, X, y, y_pred_cv



def test_random_forest(df, model):
    """
    Testing Random Forest Models.
    """
    all_targets = ['KSOL','LogD', 'log_HLM CLint', 'log_MLM CLint']

    X =df.drop(columns=all_targets).select_dtypes(include=['float64', 'int64']) #drop the target columns and only keep int64 and float64 dtypes

    y_pred = model.predict(X) #predict the targets using the features (X)

    return y_pred