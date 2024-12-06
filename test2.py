import tensorflow as tf
from PIL import Image
import numpy as np

# Chemin vers le modèle .h5
model_path = 'model/mlops_project_model.h5'
# Chemin vers l'image
image_path = 'data/image.jpg'
# Charger le modèle
model = tf.keras.models.load_model(model_path)
# Charger l'image
image = Image.open(image_path)
# Prétraitement (à adapter selon votre modèle)
image = image.resize((64, 64))  # Exemple: redimensionner à 224x224
image_array = np.array(image)
image_array = image_array / 255.0  # Normaliser les valeurs des pixels
image_array = np.expand_dims(image_array, axis=0)  # Ajouter une dimension pour le batch

# Prédiction
prediction = model.predict(image_array)

# Afficher la prédiction
print(prediction)

# Interprétation de la prédiction (à adapter selon votre modèle)
# Par exemple, si votre modèle est un classifieur d'images:
predicted_class_index = np.argmax(prediction)
# ... utiliser predicted_class_index pour obtenir le nom de la classe prédite ...