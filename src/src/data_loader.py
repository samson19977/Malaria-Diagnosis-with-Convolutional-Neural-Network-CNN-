import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm

class DataLoader:
    """Handles loading and preprocessing of malaria cell images."""
    
    def __init__(self, data_dir, img_size, test_split=0.2, random_seed=42):
        self.data_dir = data_dir
        self.img_size = img_size
        self.test_split = test_split
        self.random_seed = random_seed
        self.class_names = ['Parasitized', 'Uninfected']
        
    def load_images_from_folder(self, folder):
        """Load images from a folder and return arrays of images and labels."""
        images = []
        labels = []
        for label, subfolder in enumerate(self.class_names):
            subfolder_path = os.path.join(folder, subfolder)
            if not os.path.exists(subfolder_path):
                raise FileNotFoundError(f"Folder not found: {subfolder_path}")
            image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for filename in tqdm(image_files, desc=f"Loading {subfolder}"):
                img_path = os.path.join(subfolder_path, filename)
                try:
                    img = load_img(img_path, target_size=(self.img_size, self.img_size))
                    img = img_to_array(img) / 255.0  # Normalize to [0,1]
                    images.append(img)
                    labels.append(label)  # 0 for Parasitized, 1 for Uninfected
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
        return np.array(images), np.array(labels)
    
    def load_data(self):
        """Load all images, split into train/test, and return."""
        print("Loading dataset...")
        images, labels = self.load_images_from_folder(self.data_dir)
        print(f"Loaded {images.shape[0]} images.")
        
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=self.test_split, random_state=self.random_seed, stratify=labels
        )
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test
