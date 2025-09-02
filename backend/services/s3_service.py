import boto3
from core.config import settings

s3_client = boto3.client(
    "s3",
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_DEFAULT_REGION
)

def upload_file_obj(file_obj, bucket, key, extra_args=None):
    if extra_args is None:
        extra_args = {}

    s3_client.upload_fileobj(
        Fileobj=file_obj,
        Bucket=bucket,
        Key=key,
        ExtraArgs=extra_args
    )


def delete_file(bucket, key):
    s3_client.delete_object(Bucket=bucket, Key=key)

def generate_presigned_url(bucket, key):
    return s3_client.generate_presigned_url(
        'get_object',
        Params={
            'Bucket': bucket,
            'Key': key,
            'ResponseContentDisposition': 'inline'  
        },
        ExpiresIn=3600
    )   