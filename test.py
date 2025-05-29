import os
import numpy as np
import zipfile
import tensorflow
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

from tqdm import tqdm


base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Wrap with GlobalMaxPooling
model = tensorflow.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])
import pickle


pickle.dump(model,open('model.pkl','wb'))