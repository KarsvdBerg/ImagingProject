import os

import numpy as np
import pandas as pd
#import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from sklearn.utils import shuffle 
from sklearn.metrics import roc_curve, auc, roc_auc_score
from tqdm import tqdm_notebook

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

IMAGE_SIZE = 96


def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=64, test_batch_size=32):

     # dataset parameters
     train_path = os.path.join(base_dir, 'train+val', 'train')
     valid_path = os.path.join(base_dir, 'train+val', 'valid')
     
     RESCALING_FACTOR = 1./255

     # instantiate data generators
     datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)

     train_gen = datagen.flow_from_directory(train_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=train_batch_size,
                                             class_mode='binary')

     val_gen = datagen.flow_from_directory(valid_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=val_batch_size,
                                             class_mode='binary')
     
     val_gen_full = datagen.flow_from_directory(valid_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=16000,
                                             class_mode='binary',
                                             shuffle=False)
     
     return train_gen, val_gen, val_gen_full

train_gen, val_gen, val_gen_full = get_pcam_generators('/Users/s150055/Documents/TUe/Programming languages/Python/Imaging')

json_file = open('my_first_cnn_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('my_first_cnn_model_weights.hdf5')
print('Loaded model from disk') 

#Assigning x_val and y_val
x_val = val_gen_full[0][0]
y_val = val_gen_full[0][1]

#Testing our validation data
predictions = loaded_model.predict([x_val])

#Generate ROC
fpr, tpr, _ = roc_curve(y_val, predictions)
auc_val = auc(fpr, tpr)

# Plot ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % auc_val)
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
print('AUC: {:.3f}'.format(auc_val))
