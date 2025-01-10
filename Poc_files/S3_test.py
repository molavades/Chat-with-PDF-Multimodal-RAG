import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError

def upload_to_s3(local_file, bucket_name, s3_file):
    # Initialize S3 client
    s3 = boto3.client('s3')

    try:
        # Upload the file
        s3.upload_file(local_file, bucket_name, s3_file)
        print(f"File '{local_file}' successfully uploaded to '{bucket_name}/{s3_file}'")

    except FileNotFoundError:
        print(f"Error: The file '{local_file}' was not found.")
    
    except NoCredentialsError:
        print("Error: AWS credentials not available.")
    
    except PartialCredentialsError:
        print("Error: Incomplete credentials provided.")
    
    except ClientError as e:
        # Catch other client-related errors like permission issues
        print(f"Error: {e.response['Error']['Message']}")
    
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred: {str(e)}")

# Usage example
local_file_path = './cfa_texts/A Cash-Flow Focus for Endowments and Trusts.txt'  # Path to your local file
bucket_name = 'cfa-bigdata'  # Your S3 bucket name
s3_file_name = 'A Cash-Flow Focus for Endowments and Trusts.txt'  # Desired S3 object name

upload_to_s3(local_file_path, bucket_name, s3_file_name)