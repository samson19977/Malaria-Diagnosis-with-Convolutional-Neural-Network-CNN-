import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import logging

logger = logging.getLogger(__name__)

def predict_image(model_path, image_path, img_size):
    """Load a model and predict the class of a single image."""
    model = load_model(model_path)
    logger.info(f"Loaded model from {model_path}")
    
    # Load and preprocess image
    img = load_img(image_path, target_size=(img_size, img_size))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Predict
    pred_prob = model.predict(img_array)[0][0]
    if pred_prob > 0.5:
        return f"Uninfected (probability: {pred_prob:.4f})"
    else:
        return f"Parasitized (probability: {1-pred_prob:.4f})"
