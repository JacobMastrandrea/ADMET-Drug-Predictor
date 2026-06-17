from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score


def eval_model(y_true, y_pred):
    """
    Caluclate and return MSE, RMSE, MAE.
    """

    mse = mean_squared_error(y_true,y_pred)
    rmse = root_mean_squared_error(y_true,y_pred)
    mae = mean_absolute_error(y_true,y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return mse, rmse, mae, r2