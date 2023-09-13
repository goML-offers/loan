
import json
import boto3
from boto3.session import Session
import os
from dotenv import load_dotenv
load_dotenv()





s3_client = boto3.client('s3', aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                      aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY_ID"),
                      region_name=os.environ.get("region_name"),
                      )

# set bucket name of s3 client
bucket_name = "gomloffers"


def push_to_s3(file_path,company_name):
    print(file_path)
    folder_name = company_name
    file_name = file_path.split('/')[-1]
    print(file_name)

    s3_key = "finance_data_generator/"+folder_name+"/"+file_name
    s3_client.upload_file(Key=s3_key, Filename=file_path, Bucket=bucket_name)
    os.remove(file_path)
   
    return s3_key



def get_from_s3(s3_path):
    print("s3_path",s3_path)
    
    file_name=s3_path.split('/')[-1]
    file_name=file_name.split('uploads')[-1]
    print("file_name",file_name)
    if not file_name[0].isalpha():
        file_name= file_name[1:]
    
    print("file_name1",file_name,type(file_name))
    # with open(f'file_name', 'wb') as local_file:
    s3_client.download_file(Key=s3_path, Filename =file_name, Bucket=bucket_name)
    
    # os.remove(file_name)
    # print(file_name)
    return file_name

