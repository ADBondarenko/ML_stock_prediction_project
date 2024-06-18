from sklearn.linear_model import LinearRegression
from sklearn.metrics import (r2_score, mean_squared_error)
import pickle
import logging
import torch
from datetime import datetime
from .get_data import (get_keys)

def get_error_metrics(y_val, y_hat) -> dict:
    '''
    Validates regression quality with R2-score and RMSE. 
    Please note that the method is suitable for either 
    1-step ahead and n-step ahead forecasts
    
    Returns a dict with results

    kwargs: y_val 
            y_hat

    returns: metrics_dict -> dict
    '''
    y_val = torch.Tensor(y_val.to_numpy()).to('cpu')
    y_hat = torch.Tensor(y_hat).to('cpu')
    y_val = y_val.cpu().numpy()
    y_hat = y_hat.cpu().numpy()
    _log = logging.getLogger(__name__)
    _log.info(f"Validating prediction quality...")

    metrics_dict = {"R2_score" : float(r2_score(y_true = y_val, y_pred = y_hat)),
                    "RMSE" : float(mean_squared_error(y_true = y_val, y_pred = y_hat, squared = False))}
    _log.info(f"Got offline metrics.")
    return metrics_dict

 
#Under development
# def get_business_metrics(y_val, y_hat) ->dict:
#     '''
#     Returns some business metrics on validation dataset
#     in a 1-step ahead decision-making framework

    
#     '''

def predict(X_val, fitted_model):
    '''
    Takes a preprocessed dataset with numerous features and returns predictions on
    multipled object.

    kwargs: X_val
            fitted_model

    returns: y_hat, an array of predictions
    '''
    _log = logging.getLogger(__name__)
    
    _log.info(f"Inferring data...")
    y_hat = fitted_model.predict(X_val)
    
    _log.info(f"Inference complete!")
    return y_hat

def validate_model(X_val, y_val, fitted_model) -> dict:
    '''
    A method for pulling together all business and statistical 
    metrics together. As of current version, a simple call
    to get_error_metrics() would suffice instead.

    Returns a nested dict.

    kwargs: X_val
            y_val
            fitted_model

    returns: metrics_full : dict
    '''

    y_hat = predict(X_val, fitted_model)

    metrics_stats = get_error_metrics(y_val, y_hat)
    metrics_full = {}
    metrics_full["stats"] = {metrics_stats}

    # metrics["business"] = {}

    return metrics_full

    
    

    

#Under development since feature generation in this framework is quite a fuss
#Needs a separate prediction model for volume, open, high, low, close at least
#Also RSI, EMA calculations with forecast plugged in 

# def forecast_n_steps_ahead(fitted_model, X_val : pd.DataFrame, n : int = 5):
#     '''
#     Takes a preprocessed dataset with numerous features, takes last value 
#     and returns predictions with n-steps-ahead

#     kwargs: X_val
#             fitted_model

#     returns: y_hat, an array of predictions
#     '''

#     _log = logging.getLogger(__name__)

#     X_val = X_val.values[-1]
#     y_hat = []

#     for step_ahead in range(1, n+1)
#         y_n_hat = 
    
#     _log.info(f"Inferring data...")
#     y_hat = fitted_model.predict(X_val)
    
#     _log.info(f"Inference complete!")
#     y_hat = np.array(y_hat).reshape(-1,1)

#     return y_hat
