import boto3

# Create an S3 client
s3 = boto3.client('s3')

# Replace with your bucket name
bucket_name = 'Satellite image' 

def upload_file(file_name, object_key):
  """Uploads a file to an S3 bucket.
  

  Args:
      file_name: The path to the file to upload.
      object_key: The key (path/name) of the object in the bucket.
  """
  try:
      response = s3.upload_file(file_name, bucket_name, object_key)
      print(f"File '{file_name}' uploaded to '{bucket_name}/{object_key}'")
  except Exception as e:
      print(e)
      print(f"Error uploading '{file_name}' to '{bucket_name}/{object_key}'")

def download_file(object_key, file_name):
  """Downloads a file from an S3 bucket.

  Args:
      object_key: The key (path/name) of the object in the bucket.
      file_name: The path to save the downloaded file.
  """
  try:
      s3.download_file(bucket_name, object_key, file_name)
      print(f"File '{object_key}' downloaded to '{file_name}'")
  except Exception as e:
      print(e)
      print(f"Error downloading '{object_key}' from '{bucket_name}'")

def list_objects():
  """Lists objects in an S3 bucket."""
  try:
      response = s3.list_objects_v2(Bucket=bucket_name)
      for obj in response.get('Contents', []):
          print(f"Object Key: {obj['Key']}")
  except Exception as e:
      print(e)
      print(f"Error listing objects in bucket '{bucket_name}'")

# Example usage:
upload_file('path/to/your/image.tif', 'images/image.tif')
download_file('images/image.tif', '/tmp/downloaded_image.tif')
list_objects()