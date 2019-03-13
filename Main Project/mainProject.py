'''
TU/e BME Project Imaging 2019
Convolutional neural network for PCAM
Author: Roderick
'''

import os

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import model_from_json
# unused for now, to be used for ROC analysis
from sklearn.metrics import roc_curve, auc
import keras


# the size of the images in the PCAM dataset
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


def get_model(kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64):

     # build the model
     model = Sequential()

     model.add(Conv2D(first_filters, kernel_size, activation = 'relu', padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
     model.add(MaxPool2D(pool_size = pool_size))

     model.add(Conv2D(second_filters, kernel_size, activation = 'relu', padding = 'same'))
     model.add(MaxPool2D(pool_size = pool_size))

     model.add(Flatten())
     model.add(Dense(64, activation = 'relu'))
     model.add(Dense(1, activation = 'sigmoid'))


     # compile the model
     model.compile(SGD(lr=0.1, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])
     return model


# get the model
#model = get_model()
#model = model_from_json("my_first_cnn_model.json")
#model.load_weights("my_first_transfer_model_weights.hdf5")
model = keras.engine.saving.load_model(r"C:\\Users\\stoer\\Imaging\\results.hdf5")
model.compile(SGD(lr=0.1, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])

# get the data generators
train_gen, val_gen = get_pcam_generators(r'C:\Users\stoer\Imaging')


# save the model and weights
model_name = 'my_first_cnn_model'
model_filepath = model_name + '.json'
weights_filepath = model_name + '_weights.hdf5'

#

model_json = model.to_json() # serialize model to JSON
with open(model_filepath, 'w') as json_file:
    json_file.write(model_json)


# define the model checkpoint and Tensorboard callbacks
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(os.path.join('logs', model_name))
callbacks_list = [checkpoint, tensorboard]


# train the model
train_steps = train_gen.n//train_gen.batch_size
val_steps = val_gen.n//val_gen.batch_size

#history = model.fit_generator(train_gen, steps_per_epoch=train_steps,
#                    validation_data=val_gen,
#                    validation_steps=val_steps,
#                    epochs=3,
#                    callbacks=callbacks_list)

keras.engine.saving.save_model(model, r"C:\\Users\\stoer\\Imaging\\results.hdf5", overwrite=True, include_optimizer=True)
# ROC analysis
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
#roc_curve()
# TODO Perform ROC analysis on the validation set

"""
