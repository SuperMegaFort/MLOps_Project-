# Satellite Image Prediction with Google Cloud Functions

This project uses a Google Cloud Function to predict features in satellite images uploaded to a Cloud Storage bucket. It leverages TensorFlow/Keras for model inference and stores predictions in a CSV file and organizes predicted images into folders within the same bucket.

## Prerequisites

Before deploying this project, ensure you have the following:

*   **Google Cloud Project:** A Google Cloud project with billing enabled.
*   **gcloud CLI:** The Google Cloud command-line interface installed and configured.
*   **Python 3:** Python 3 installed on your local machine.
*   **Required Python Libraries:** Install the necessary Python libraries using pip:
 

## Project Structure
.
├── data
│   ├── raw
│   │   └── ...
│   └── README.md
├── src 
│   ├── utils
│   │   ├── __init__.py
│   │   └── seed.py
│   ├── evaluate.py
│   ├── prepare.py
│   └── train.py
├── README.md 
├── params.yaml 
├── requirements-freeze.txt 
└── requirements.txt 

Cloud Function

### Deployment Steps
Create Cloud Storage Buckets:

Create three Cloud Storage buckets:
- nut-bucket: For uploading satellite images to be processed.
- tensorflow-mlops: For storing the trained TensorFlow/Keras model (mlops_project_model.h5).
- output-predictions: For storing the prediction CSV file.
Important: the output image will be save in the same bucket of the input image in a folder named "predictions"

```bash
- gsutil mb gs://input-bucket
- gsutil mb gs://tensorflow-mlops
- gsutil mb gs://output-predictions
```
### Upload Model:

Upload your trained TensorFlow/Keras model (mlops_project_model.h5) to the tensorflow-mlops bucket.

```bash
gsutil cp mlops_project_model.h5 gs://tensorflow-mlops/
```
### Deploy the Cloud Function:

Navigate to the directory containing main.py.
Deploy the Cloud Function using the gcloud command:
Bash
```bash
gcloud functions deploy analyser_image \
--runtime python39 \
--trigger-resource gs://input-bucket \
--trigger-event google.storage.object.finalize \
--source . \
--region <YOUR_REGION> \
--timeout=540s \
--memory=512MB  
```
### Test the Function:

Upload an image to the input-bucket.
Check the Cloud Function logs in the Google Cloud Console to monitor the prediction process.
Verify that the prediction CSV file (predictions.csv) is created in the output-predictions bucket and that predicted images are saved in the "predictions" folder of the input bucket, organized by prediction labels.
