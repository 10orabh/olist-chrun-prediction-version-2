import os
import boto3
from dotenv import load_dotenv
from utils.logger import Logger
from botocore.exceptions import ClientError

load_dotenv()
logger = Logger('s3_utils', level="DEBUG").get_logger()

class S3Connector:
    def __init__(self):
        self.aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.endpoint_url = os.getenv("AWS_ENDPOINT_URL")
        self.region_name = os.getenv("AWS_REGION", "us-east-1")
        self.s3_client = self._create_s3_client()

    def _create_s3_client(self):
        try:
            logger.info("Connecting to AWS S3...")
            client = boto3.client(
                's3',
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.region_name
            )
            logger.info("S3 Connection established.")
            return client
        except Exception as e:
            logger.error(f"S3 Connection failed: {e}")
            raise

    def upload_file(self, local_path, bucket, s3_key):
        try:
            logger.info(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
            self.s3_client.upload_file(local_path, bucket, s3_key)
            logger.info("Upload successful.")
        except ClientError as e:
            logger.error(f"Upload failed: {e}")
            return False
        return True

    def download_file(self, bucket, s3_key, local_path):
        try:
            logger.info(f"Downloading s3://{bucket}/{s3_key} to {local_path}")
            self.s3_client.download_file(bucket, s3_key, local_path)
            logger.info("Download successful.")
        except ClientError as e:
            logger.error(f"Download failed: {e}")
            return False
        return True