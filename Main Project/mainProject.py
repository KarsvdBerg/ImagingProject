# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 19:45:05 2019

@author: stoer
"""

#val_gen & train_gen
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard

import keras
#from keras import save_model


#model_filepath = r"C:\Users\stoer\Imaging\results.py"
#keras.engine.saving.save_model(model, r"C:\\Users\\stoer\\Imaging\\results.hdf5", overwrite=True, include_optimizer=True)
model = keras.engine.saving.load_model(r"C:\\Users\\stoer\\Imaging\\results.hdf5")
IMAGE_SIZE = 96

def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):

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

     return train_gen, val_gen

train_gen, val_gen = get_pcam_generators(r'C:\Users\stoer\Imaging')

#model = model_from_json("my_first_cnn_model.json")
model.compile(SGD(lr=0.1, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])
#model.load_weights("my_first_transfer_model_weights.hdf5")

train_steps = train_gen.n//train_gen.batch_size
val_steps = val_gen.n//val_gen.batch_size
"""
a = val_gen.__dict__
trueList = []
falseList = []
truefalseindex = val_gen.classes
for x in range(len(val_gen.filenames)):
    if truefalseindex[x]==0:
        falseList.append(val_gen.filenames[x])
    else:
        trueList.append(val_gen.filenames[x])

file = open('val_gen.txt','w')
file.write(str(trueList))
file.close()
"""
