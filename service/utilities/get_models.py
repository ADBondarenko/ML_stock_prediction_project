import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from .get_data import (get_keys)
import logging
import pickle
def get_models_by_ticker_timeframe(ticker : str, timeframe : str): 
    '''
    '''
    _log = logging.getLogger(__name__)
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

    #List all models by ticker and timeframe 
    bucket_name = "ml-project"
    
    objects = s3.list_objects(Bucket='ml-project')["Contents"]
    _log.info(f"Getting models info from S3...")
    key_list = []
    for object in objects:
        key = object["Key"]
        key_tokens = key.split("_")
        if (ticker == key_tokens[0]) and (timeframe ==key_tokens[2]):
            key_list.append(key)     
    else:
        pass

    _log.info(f"Info downloaded from S3.")
    
    _log.info(f"Pulling models from S3...")
    models_list = []
    if len(models_list) > 0: 
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
    API_S3_SECRET = keychain["API_S3_SECRET"]

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
    
    objects = s3.list_objects(Bucket='ml-project')["Contents"]
    _log.info(f"Getting models info from S3...")
    key_list = []
    for object in objects:
        key = object["Key"]
        key_tokens = key.split("_")
        if (ticker == key_tokens[0]):
            key_list.append(key)     
        else:
            pass

    _log.info(f"Info downloaded from S3.")
    
    _log.info(f"Pulling models from S3...")
    models_list = []
    if len(key_list) > 0: 
        for model_key in key_list:
            response = s3.get_object(Bucket='ml-project', Key=model_key)
            file_ = response["Body"].read()
            models_list.append(file_)
        _log.info(f"Models succesfully pulled")
    else: 
        _log.info(f"Nothing to pull!")
    

    return key_list #model_list will encounted the same proble as with bytestring pickle objects

def get_model(model_name : str): 
    '''
    '''
    _log = logging.getLogger(__name__)
    model_name = f"{model_name}.pkl"
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

    #List all models by ticker and timeframe 
    bucket_name = "ml-project"
    
    response = s3.get_object(Bucket='ml-project', Key=model_name)
    file_ = response["Body"].read()
    _log.info(f"Model succesfully pulled!")

    return file_

def get_all_models(most_recent : bool = True): 
    '''
    By default - pulls the name of most_recent (last page of an S3 trained)
    models from S3. If most_recent is set to False - pulls 
    
    '''
    _log = logging.getLogger(__name__)
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
    paginator = s3_client.get_paginator('list_objects_v2')
    bucket_name = "ml-project"
    models_list = []
    if most_recent == True:
        for page in paginator.paginate(Bucket=bucket_name):
            for 'Contents' in page:
                for obj in page['Contents']:
                    models_list.append(obj['Key'])
            break
    else: 
        for page in paginator.paginate(Bucket=bucket_name):
            for 'Contents' in page:
                for obj in page['Contents']:
                    models_list.append(obj['Key'])
    
    _log.info(f"Model succesfully pulled!")

    return models_list
