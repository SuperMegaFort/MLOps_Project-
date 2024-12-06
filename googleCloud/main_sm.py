import os
from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

# Charger le modèle au démarrage de la fonction
storage_client = storage.Client()
bucket = storage_client.bucket('tensorflow-mlops')  # Remplacez par le nom de votre bucket
model_folder = 'mlops_project_model.savedmodel'  # Nom du dossier du modèle
local_model_path = f'/tmp/{model_folder}'

blobs = bucket.list_blobs(prefix=model_folder)
for blob in blobs:
    file_path = os.path.join(local_model_path, os.path.basename(blob.name))
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    blob.download_to_filename(file_path)

modele = tf.saved_model.load(local_model_path)

def analyser_image(event, context):
    """Fonction cloud déclenchée par l'ajout d'une image dans un bucket."""
    file = event
    bucket_name = file['bucket']
    file_name = file['name']

    # Télécharger l'image depuis Cloud Storage
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    image_path = f'/tmp/{file_name}'
    blob.download_to_filename(image_path)

    # Prétraiter l'image
    img = Image.open(image_path)  
    img = img.resize((64, 64))  # Adaptez la taille si nécessaire
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0) 

    # Inférence avec le modèle
    prediction = model(img_array)  

    # Post-traitement et interprétation de la prédiction
    predicted_class_index = np.argmax(prediction)
    with open('labels.json') as f:  # Assurez-vous que labels.json est accessible
        labels = json.load(f)
    predicted_class_name = labels[predicted_class_index]

    print(f"Image {file_name}: classe prédite = {predicted_class_name}")