import os
import numpy as np
import zipfile
import tensorflow
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tqdm import tqdm
import pickle

# Load ResNet50 model without top layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Wrap with GlobalMaxPooling
model = tensorflow.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Feature extraction function
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Path to ZIP file
zip_path = r'C:\Users\VARDAN\Downloads\archive.zip'

# Extract ZIP to a temp folder
extract_path = 'temp_images'
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Path to extracted images folder
image_folder = os.path.join(extract_path, 'images')

# Get all valid image paths
image_paths = [os.path.join(image_folder, fname)
               for fname in os.listdir(image_folder)
               if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Extract features
features_list = []
for path  in tqdm(image_paths):
    features = extract_features(path, model)
    features_list.append(features)

# Convert to NumPy array
pickle.dump(features_list,open('embeddings.pkl','wb'))
pickle.dump(image_paths,open('filenames.pkl','wb'))