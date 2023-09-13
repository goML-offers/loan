import os
import subprocess
import boto3
from boto3.session import Session


s3_object_keys = ['procurement/wheel_files/boto3-1.28.21-py3-none-any.whl', 'procurement/wheel_files/botocore-1.31.21-py3-none-any.whl']


AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID_S3")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY_S3")



session = Session(aws_access_key_id=AWS_ACCESS_KEY_ID,
aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
s3 = session.resource('s3')
s3_client = s3.Bucket('gomloffers')


def download_wheel_from_s3( object_key, local_path):

    # s3 = boto3.client('s3')

    print(object_key.split('/')[-1])
    s3_client.download_file(Key=object_key, Filename =object_key.split('/')[-1])

def install_wheel(wheel_file_path):

    subprocess.run(['pip', 'install', wheel_file_path])

def main():

    local_download_path = '/'

    for s3_object_key in s3_object_keys:

        download_wheel_from_s3(s3_object_key, local_download_path)


        install_wheel(os.path.join(local_download_path, s3_object_key))

if __name__ == "__main__":
    main()