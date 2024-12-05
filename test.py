from PIL import Image
import numpy as np
import bentoml

# Load the BentoML model
model = bentoml.keras.load_model("mobilenetv2_modified_classifier_model:latest")  # Replace with your model name

# Load and preprocess the image
image = Image.open("image.jpg")  # Replace with your image path
image = image.resize((64, 64))  # Resize to match your model's input size
image = np.array(image)
image = image / 255.0  # Normalize if necessary
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Run inference
predictions = model.predict(image)

# Handle the prediction
predicted_class = np.argmax(predictions)  # Get the class with the highest probability
print(f"Predicted class: {predicted_class}")