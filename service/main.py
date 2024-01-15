from enum import Enum
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import os
import psycopg2
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

from .utilities.get_data import (get_keys, get_tickers, get_history_by_ticker)
from .utilities.get_models import (get_models_by_ticker_timeframe, get_models_by_ticker, get_model)
from .utilities.generate_features import (gen_re_rsi, gen_re_rsi_mtf, gen_rsi, gen_ema, generate_features, cleanup_and_prepare_data)
from .utilities.train_model import (train_model)
from .utilities.predict_validate import (get_error_metrics, predict, validate_model)

#Inititalize app
ml_app = FastAPI()

#Create classes for Requests methods
class ModelType(str, Enum):
    random_forest = "rf"
    linear_regression = "lr"
    hist_gradient_boosting = "hgb"
   
class Timeframe(str, Enum):
    m1 = "1m"
    m3 = "3m"
    m5 = "5m"
    m15 = "15m"
    m30 = "30m"
    m60 = "1H"
    m120 = "2H"
    m240 = "4H"
    m360 = "6H"
    m1440 = "1D"
    m5320 = "3D"
    m10080 = "1W"

@ml_app.get('/',
         summary = "Root",
         operation_id =  "root__get", 
         response_model = {})
async def root() -> dict:
    return {"Status" : "Prediction service is online."}
    

@ml_app.get('/model/get_models_by_ticker', 
            operation_id = "get__model__models_by_ticker",
            summary = "Get the best availible model with predictions, its metrics or create a new one")
async def get_models_by_ticker(ticker : str):
    model_list = get_models_by_ticker(ticker)

    return {"models" : model_list}

@ml_app.get('/model/get_model_by_name', 
            operation_id = "get__model__model_by_name",
            summary = "Returns a Pickle-serialized sklearn trained model object")
async def get_model_by_name(model_name : str):
    model_pkl = get_model(model_name)

    return {"model" : model_pkl}
    

    
@ml_app.get('/model/get_new_model', 
            operation_id = "get__model__new_model",
            summary = "Trains a new model then returns its metrics on server-side along with a handle")
async def get_new_model(ticker : str, timeframe : Timeframe, model_type : ModelType, num_bars_back : int, binary_rsi : bool = True,
                            rsi_period : int =14 , rsi_levels : list = [20,40,60,80], binary_ema : bool = True,
                            ema_periods : list = [8,24], nth_diff : int = 1):
    #Getting data
    data_raw = get_history_by_ticker(ticker, timeframe, num_bars_back)
    #Preprocessing
    data = generate_features(data_raw, binary_rsi, binary_ema, nth_diff,
                      rsi_period, rsi_levels, ema_periods)
    training_data, columns = cleanup_and_prepare_data(data)
    
    X_train, y_train, X_val, y_val = training_data
    #Training model
    new_model, new_model_handle = train_model(X_train, y_train, ticker, timeframe, model_type)
    y_hat_train = predict(X_train, new_model)
    y_hat_val = predict(X_train, new_model)
    #Getting metrics
    metrics_train = get_error_metrics(y_train, y_hat_train)
    metrics_val = get_error_metrics(y_val, y_hat_val)
    return {"model_name" : new_model_handle,
            "metrics_train" : metrics_train,
            "metrics_val" : metrics_val}
    


@ml_app.get('/model/predict', 
            operation_id = "get__model__predict",
            summary = "Pass a model name to get a prediction on server_side along with metrics")   
async def predict(model_name : str, ticker : str, timeframe : Timeframe, num_bars : int = 40):

    #Download model from S3
    model_pkl = get_model(model_name)
    model = pickle.loads(model)

    #Get data for prediction
    data_raw = get_history_by_ticker(ticker, timeframe, num_bars_back)
    data = generate_features(data_raw)
    
    training_data, columns = cleanup_and_prepare_data(data)
    
    X_train, y_train, X_val, y_val = training_data

    y_hat = predict(X_val, model)

    metrics = get_error_metrics(y_val, y_hat)

    return {"y_hat" : y_hat, "validation_matrics" : metrics}
    
    
    
    
    
    

    

    


















                            
                            
@app.get('/dog',
        summary = "Get Dogs",
        operation_id =  "get_dogs_dog_get", 
        response_model = Dog)
async def get_dog(kind : DogType | None = None) -> list[Dog]:
    
    response = []
    
    if kind == None:
        response.append(dogs_db)
        
    else:
        for dog in list(dogs_db.values()):
        
            if dog.kind == kind:
                            
                response.append(dog)
             
    return response
    
    
@app.post('/dog',
        summary = "Create Dog",
        operation_id =  "create_dog_dog_post", 
        response_model = Dog)
async def post_dog(name : str, pk: int, kind : DogType) -> Dog:
    
    
    new_dog_name = name
    new_dog_pk = pk
    new_dog_kind = DogType(kind)

    new_dog = Dog(name = new_dog_name, pk = new_dog_pk, kind = new_dog_kind)
    if new_dog.pk in dogs_db.keys():
        
        raise HTTPException(status_code = 422, detail = "Dog with a given 'pk' already exists")
        
    else: 
        dogs_db[pk] = new_dog
        # reponse = {
        #     "name" : name,
        #     "pk" : pk,
        #     "kind" : kind                  
        # }
        response = new_dog
        
        return response
    