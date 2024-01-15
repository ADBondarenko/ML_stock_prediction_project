from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (RandomForestRegressor, HistGradientBoostingRegressor)
import pickle
import logging
from datetime import datetime
from .get_data import (get_keys)
import boto3 

def train_model(X_train, y_train, ticker : str, timeframe : str, model : str = "rf"):
    '''
    Takes a preprocessed dataset with numerous features, trains model, 
    saves it to an S3 remote storage, then returns a trained model instance
    '''
    _log = logging.getLogger(__name__)
    
    implemented_models_dict = {"lr" : LinearRegression, 
                               "rf" : RandomForestRegressor,
                               "hgb" : HistGradientBoostingRegressor}

    if model not in implemented_models_dict.keys():
        raise ValueError("Model not implemented yet")

    cur_model = implemented_models_dict[model]()
    now = datetime.now()
    _log.info(f"Started training model at {now} timestamp")
    cur_model.fit(X_train, y_train)
    then = datetime.now()
    time_diff_seconds = (then - now).total_seconds()
    _log.info(f"Model succesfully trained in {time_diff_seconds} s.")
    model_name = f"{ticker}_{model}_{timeframe}_{then}_train"
    
    
    now = datetime.now()
    _log.info(f"Started saving model at timestamp {now}...")
    

    #Getting keys
    keychain = get_keys()
    API_S3_ID = keychain["API_S3_ID"]
    API_S3_SECRET = keychain["API_S3_SECRET"]

    #Opening client
    session = boto3.session.Session()
    s3 = session.client(
    service_name='s3',
    endpoint_url='https://storage.yandexcloud.net',
    region_name = 'ru-central1',
    aws_access_key_id = API_S3_ID,
    aws_secret_access_key = API_S3_SECRET)
    
    #Pickled model
    model_pickled = pickle.dumps(cur_model)

    #Upload to S3
    bucket_name = "ml-project"
    key = f"{model_name}.pkl"
    body = model_pickled
    s3.put_object(Bucket="ml-project", Key=key, Body=model_pickled,
                Metadata = {
              "ticker" : ticker,
              "model" : model_name,
              "created_at" : f"{then}"})
    
    _log.info(f"Model succesfully saved")
    _log.info(f"Model deleted from local storage")
    
    return cur_model, model_name


    