from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (RandomForestRegressor, HistGradientBoostingRegressor)
from .dl_models import MLPModel, create_sequences
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import pickle
import logging
from datetime import datetime
from .get_data import (get_keys)
import boto3 
import torch.nn as nn

def train_model(X_train, y_train, ticker : str, timeframe : str, model : str = "rf"):
    '''
    Takes a preprocessed dataset with numerous features, trains model, 
    saves it to an S3 remote storage, then returns a trained model instance
    '''
    _log = logging.getLogger(__name__)
    
    implemented_models_dict = {"lr" : LinearRegression, 
                               "rf" : RandomForestRegressor,
                               "hgb" : HistGradientBoostingRegressor,
                              "mlp" : MLPModel}

    if model not in implemented_models_dict.keys():
        raise ValueError("Model not implemented yet")
        
    if model == "lstm":
        pass
        # #Архитектура выбрана на базе DOI: 10.1142/S0129065721300011
        # #Гиперпараметры прибиты гвоздями для быстродействия
        # batch_size = 32   
        # time_steps = 5    # Размер скользящего окна
        # learning_rate = 0.001
        # num_epochs = 50


        
        # X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
        # X_train, y_train = create_sequences(X_train, y_train, time_steps)
        
        # X_train = torch.tensor(X_train, dtype=torch.float32)
        # y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        
        # cur_model = LSTMModel(input_size = X_train.shape[1], hidden_size = X_train.shape[1]*3, output_size = 1, num_layers=1
        #                 )
        
        # criterion = nn.MSELoss()
        # optimizer = optim.Adam(cur_model.parameters(), lr=learning_rate)
        # train_dataset = TensorDataset(X_train, y_train)
        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) #можно и так, и так, но классика для TS - не перемешивать
        # now = datetime.now()
        # _log.info(f"Started training model at {now} timestamp")
        # for epoch in range(num_epochs):
        #     cur_model.train()
        #     for i, (inputs, labels) in enumerate(train_loader):
        #         outputs = cur_model(inputs)
        #         loss = criterion(outputs, labels)
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #     now = datetime.now()
        
        # then = datetime.now()
            
        # time_diff_seconds = (then - now).total_seconds()
        # then = then.strftime("%Y-%m-%d-%H-%M-%S")
            
        # _log.info(f"Model succesfully trained in {time_diff_seconds} s.")
        # model_name = f"{ticker}_{model}_{timeframe}_{then}_train"


    elif model == "mlp":
        cur_model = MLPModel(input_size = X_train.shape[1], hidden_size = X_train.shape[1]*3, output_size = 1)
        batch_size = 32   
        learning_rate = 0.001
        num_epochs = 50
        X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
        y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(cur_model.parameters(), lr=learning_rate)
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        
        now = datetime.now()
        _log.info(f"Started training model at {now} timestamp")
        for epoch in range(num_epochs):
            cur_model.train()
            for i, (inputs, labels) in enumerate(train_loader):
                outputs = cur_model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        
        then = datetime.now()
            
        time_diff_seconds = (then - now).total_seconds()
        then = then.strftime("%Y-%m-%d-%H-%M-%S")
            
        _log.info(f"Model succesfully trained in {time_diff_seconds} s.")
        model_name = f"{ticker}_{model}_{timeframe}_{then}_train"
    else:
        cur_model = implemented_models_dict[model]()
        now = datetime.now()
        _log.info(f"Started training model at {now} timestamp")
        cur_model.fit(X_train, y_train)
        then = datetime.now()
        
        time_diff_seconds = (then - now).total_seconds()
        then = then.strftime("%Y-%m-%d-%H-%M-%S")
        
        _log.info(f"Model succesfully trained in {time_diff_seconds} s.")
        model_name = f"{ticker}_{model}_{timeframe}_{then}_train"
        
        
        
    

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
    now = datetime.now()
    _log.info(f"Started saving model at timestamp {now}...")
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


    