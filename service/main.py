from enum import Enum
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import os
import psycopg2
import boto3
from botocore.exceptions import NoCredentialsError, ClientError




session = boto3.session.Session()
s3 = session.client(
    service_name='s3',
    endpoint_url='https://storage.yandexcloud.net'
)

# Creating a new bucket
s3.create_bucket(Bucket='bucket-name')

# Uploading objects into the bucket

## From a string
s3.put_object(Bucket='bucket-name', Key='object_name', Body='TEST', StorageClass='COLD')

## From a file
s3.upload_file('this_script.py', 'bucket-name', 'py_script.py')
s3.upload_file('this_script.py', 'bucket-name', 'script/py_script.py')

# Getting a list of objects in the bucket
for key in s3.list_objects(Bucket='bucket-name')['Contents']:
    print(key['Key'])

# Deleting multiple objects
forDeletion = [{'Key':'object_name'}, {'Key':'script/py_script.py'}]
response = s3.delete_objects(Bucket='bucket-name', Delete={'Objects': forDeletion})

# Retrieving an object
get_object_response = s3.get_object(Bucket='bucket-name',Key='py_script.py')
print(get_object_response['Body'].read())

# database_connector = psycopg2.connect()
ml_app = FastAPI()


class DogType(str, Enum):
    terrier = "terrier"
    bulldog = "bulldog"
    dalmatian = "dalmatian"


class Dog(BaseModel):
    name: str
    pk: int
    kind: DogType


class Timestamp(BaseModel):
    id: int
    timestamp: int


dogs_db = {
    0: Dog(name='Bob', pk=0, kind='terrier'),
    1: Dog(name='Marli', pk=1, kind="bulldog"),
    2: Dog(name='Snoopy', pk=2, kind='dalmatian'),
    3: Dog(name='Rex', pk=3, kind='dalmatian'),
    4: Dog(name='Pongo', pk=4, kind='dalmatian'),
    5: Dog(name='Tillman', pk=5, kind='bulldog'),
    6: Dog(name='Uga', pk=6, kind='bulldog')
}

post_db = [
    Timestamp(id=0, timestamp=12),
    Timestamp(id=1, timestamp=10)
]
summary="Create an item",
description="Create an item with all the information, name, description, price, tax and a set of unique tags"

@app.get('/',
         summary = "Root",
         operation_id =  "root__get", 
         response_model = {})
async def root() -> dict:
    return {"Status" : "Service is operational."}
    
@app.post('/post',
         summary = "Get Post",
         operation_id =  "get_post_post_post", 
         response_model = Timestamp)
async def post() -> Timestamp:
    new_id = post_db[-1].id + 1
    timestamp = time.time()
    
    # response = {"id" : new_id, "timestamp" : timestamp}

    response = Timestamp(id = new_id, timestamp = timestamp)
    return response
    ...

...
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
    
@app.get('/dog/{pk}',
        summary = "Get Dog By Pk",
        operation_id =  "get_dog_by_pk_dog__pk__get", 
        response_model = Dog)
async def get_dog_by_pk(pk : int) -> Dog:
    
    if pk not in dogs_db.keys():
        
        raise HTTPException(status_code = 404, detail = "Dog with a given 'pk' is not found")
    
    else:
        dog_by_pk = dogs_db[dog_pk]
    
        # reponse = {
        #     "name" : dog_by_pk.name,
        #     "pk" : dog_by_pk.pk,
        #     "kind" : dog_by_pk.kind                  
        # }
        response = dog_by_pk
        
        return response
   
    
    
    
@app.patch('/dog/{pk}',
          summary = "Update Dog",
          operation_id =  "update_dog_dog__pk__patch", 
          response_model = Dog)
async def update_dog_by_pk(pk : int, name : str, kind = DogType) -> Dog:
    
    if pk not in dogs_db.keys():
        
        raise HTTPException(status_code = 404, detail = "Dog with a given 'pk' is not found")
        
    else: 
        
        dogs_db[pk].name = name
        dogs_db[pk].kind = kind
            
        updated_dog = dogs_db[pk]
        # response = {
        #     "name" = updated_dog.name,
        #     "pk" = updated_dog.pk,
        #     "kind" = updated_dog.kind
        # }
        response = updated_dog
        return response