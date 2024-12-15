import os
import logging
from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np
from cloudevents.http import CloudEvent
import functions_framework

# Configure logging to display messages in the function logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

@functions_framework.cloud_event
def analyser_image(cloud_event: CloudEvent):
    logger.info("Function triggered by a Cloud event.")
    
    # Get information about the triggering bucket and file
    storage_client = storage.Client()
    file = cloud_event.data
    bucket_name = file['bucket']
    file_name = file['name']

    # Download the image from the bucket
    logger.info(f"Downloading image from bucket {bucket_name}.")
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    image_path = f'/tmp/{file_name}' 
    blob.download_to_filename(image_path)

    # Preprocess the image
    logger.info(f"Preprocessing image: {file_name}.")
    try:
        img = Image.open(image_path)
        img = img.resize((64, 64))  
        img_array = np.array(img) / 255.0 
        img_array = np.expand_dims(img_array, axis=0)
    except Exception as e:
        logger.error(f"Error during image preprocessing: {e}")
        return f"Error during image preprocessing: {e}", 500

    # Prediction with the model
    try:
        logger.info(f"Starting prediction for image {file_name}.")

        # Download the model from the model bucket
        bucket = storage_client.bucket('tensorflow-mlops') 
        model_file = 'mlops_project_model.h5'
        local_model_path = f'/tmp/{model_file}'
        logger.info("Downloading model from bucket.")
        blob = bucket.blob(model_file)
        blob.download_to_filename(local_model_path)
        logger.info("Model downloaded.")

        # Load the model
        modele = tf.keras.models.load_model(local_model_path)

        # Predict the image
        prediction = modele.predict(img_array)
        prediction_class = np.argmax(prediction)  
        prediction_str = str(prediction_class)  #

        logger.info(f"Prediction complete: {prediction_str}")

        # Save the prediction to a CSV file (append mode)
        output_file_name = "predictions.csv"  
        output_bucket_name = "output-predictions"
        output_blob = storage_client.bucket(output_bucket_name).blob(output_file_name)

        # Check if the CSV file exists in the output bucket
        if output_blob.exists():
            # If it exists, download it, append the new prediction, and upload it back
            content = output_blob.download_as_string().decode('utf-8') 
            content += f"\n{file_name},{prediction_str}" 
        else:
            # If it doesn't exist, create the header and the first entry
            content = f"image_name,prediction\n{file_name},{prediction_str}"

        output_blob.upload_from_string(content.encode('utf-8')) 
        # Save the predicted image in a folder corresponding to the prediction
        predicted_folder = f"predictions/{prediction_class}"  
        output_image_blob = storage_client.bucket(bucket_name).blob(predicted_folder + "/" + file_name) 
        output_image_blob.upload_from_filename(image_path)

        return "Prediction completed successfully.", 200
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return f"Error during prediction: {e}", 500