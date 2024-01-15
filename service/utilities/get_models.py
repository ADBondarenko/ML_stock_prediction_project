import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from utilities.get_data import (get_keys)
import logging
import pickle
def get_models_by_ticker_timeframe(ticker : str, timeframe : str): 
    '''
    '''
    _log = logging.getLogger(__name__)
    #Getting keys
    keychain = get_keys()
    API_S3_ID = keychain["API_S3_ID"]
    API_S3_SECRET = keycahin["API_S3_SECRET"]

    #Opening client
    session = boto3.session.Session()
    s3 = session.client(
    service_name='s3',
    endpoint_url='https://storage.yandexcloud.net',
    region_name = 'ru-central1',
    aws_access_key_id = API_S3_ID,
    aws_secret_access_key = API_S3_SECRET)

    #List all models by ticker and timeframe 
    bucket_name = "ml-project"
    
    objects = s3.list_objects(Bucket='ml-project')["contents"]
    _log.info(f"Getting models info from S3...")
    key_list = []
    for object in objects:
        key = object["key"]
        key_tokens = key.split("_")
        if (ticker == key_tokens[0]) and (timeframe ==key_tokens[2]):
            key_list.append(key)     
    else pass

    _log.info(f"Info downloaded from S3.")
    
    _log.info(f"Pulling models from S3...")
    models_list = []
    if len(model_list) > 0: 
        for model_key in models_list:
            response = s3.get_object(Bucket='ml-project', Key=model_key)
            file_ = response["Body"].read()
            model_ = pickle.loads(file_)
            models_list.append(model_)
        _log.info(f"Models succesfully pulled")
    else: 
        _log.info(f"Nothing to pull!")
    

    return models_list

def get_models_by_ticker(ticker : str): 
    '''
    '''
    _log = logging.getLogger(__name__)
    #Getting keys
    keychain = get_keys()
    API_S3_ID = keychain["API_S3_ID"]
    API_S3_SECRET = keycahin["API_S3_SECRET"]

    #Opening client
    session = boto3.session.Session()
    s3 = session.client(
    service_name='s3',
    endpoint_url='https://storage.yandexcloud.net',
    region_name = 'ru-central1',
    aws_access_key_id = API_S3_ID,
    aws_secret_access_key = API_S3_SECRET)

    #List all models by ticker and timeframe 
    bucket_name = "ml-project"
    
    objects = s3.list_objects(Bucket='ml-project')["contents"]
    _log.info(f"Getting models info from S3...")
    key_list = []
    for object in objects:
        key = object["key"]
        key_tokens = key.split("_")
        if (ticker == key_tokens[0]):
            key_list.append(key)     
    else pass

    _log.info(f"Info downloaded from S3.")
    
    _log.info(f"Pulling models from S3...")
    models_list = []
    if len(model_list) > 0: 
        for model_key in models_list:
            response = s3.get_object(Bucket='ml-project', Key=model_key)
            file_ = response["Body"].read()
            model_ = pickle.loads(file_)
            models_list.append(model_)
        _log.info(f"Models succesfully pulled")
    else: 
        _log.info(f"Nothing to pull!")
    

    return models_list 

def get_model(model_name : str): 
    '''
    '''
    _log = logging.getLogger(__name__)
    #Getting keys
    keychain = get_keys()
    API_S3_ID = keychain["API_S3_ID"]
    API_S3_SECRET = keycahin["API_S3_SECRET"]

    #Opening client
    session = boto3.session.Session()
    s3 = session.client(
    service_name='s3',
    endpoint_url='https://storage.yandexcloud.net',
    region_name = 'ru-central1',
    aws_access_key_id = API_S3_ID,
    aws_secret_access_key = API_S3_SECRET)

    #List all models by ticker and timeframe 
    bucket_name = "ml-project"
    
    response = s3.get_object(Bucket='ml-project', Key=model_name)
    file_ = response["Body"].read()
    _log.info(f"Model succesfully pulled!")

    return file_
