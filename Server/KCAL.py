from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import regularizers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras import models
from tensorflow.keras.applications.inception_v3 import preprocess_input
import cv2
import os
import random
import collections
from collections import defaultdict
from shutil import copy
from shutil import copytree, rmtree
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import json
import pandas as pd
from PIL import Image

def predict_class(model, img, show = True):
  img = img.convert('RGB')
  img = img.resize((299, 299))
  img = image.img_to_array(img)                    
  img = np.expand_dims(img, axis=0)         
  img = preprocess_input(img)                                      

  pred = model.predict(img)
  index = np.argmax(pred)
  with open("Diplom_Calorific\Server\model\FoodKcal.json",'r',encoding='utf-8') as f:
        d = json.loads(f.read())
        x = d[str(index)]
  return(x)

def main(img):
    model_best = load_model('Diplom_Calorific\Server\model\\trainedmodel_101class.hdf5',compile = False)
    x= predict_class(model_best, img, True)
    return(x)