import os

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard
import keras


# the size of the images in the PCAM dataset
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
     
     return train_gen, val_gen


def get_model(lr, kernel_size=(3,3), pool_size=(2,2)):

     # build the model
     model = Sequential()

     model.add(Conv2D(16, kernel_size, activation = 'relu', padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
     model.add(MaxPool2D(pool_size = pool_size))

     model.add(Conv2D(32, (2,2), activation = 'relu', padding = 'same'))
     model.add(MaxPool2D(pool_size = pool_size))
     
     model.add(Conv2D(64, (2,2), activation = 'relu', padding = 'same'))
     model.add(MaxPool2D(pool_size = pool_size))
     
     model.add(Conv2D(64, (2,2), activation = 'relu', padding = 'same'))
     model.add(MaxPool2D(pool_size = (3,3)))
       
     model.add(Flatten())
     model.add(Dense(128, activation = 'relu'))
     model.add(Dense(64, activation = 'relu'))
     model.add(Dense(1, activation = 'sigmoid'))

     # compile the model
     model.compile(SGD(lr, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])

     return model

# Export data to .csv file  
def ExportData(VALlossWhileTraining, VALaccuracyWhileTraining, lossWhileTraining, accuracyWhileTraining, lr):
    filename = "NewModelOutput_best" + str(lr)[2:] + ".csv"
    file = open(filename, "w")
    file.write("Validation Loss" + ","+ "Validation Accuracy" + "," + "loss" + "," + "Accuracy" + "," + "Epoch" + "\n")
    for x in range(len(VALlossWhileTraining[0])):
        string = str(VALlossWhileTraining[0][x]) + "," + str(VALaccuracyWhileTraining[0][x]) + "," + str(lossWhileTraining[0][x]) + "," + str(accuracyWhileTraining[0][x]) + "," + str(x+1) + "\n"
        file.write(string)
    file.close()


# get the model
## Loop over different learning rates    
learningRate = [0.1]
epoch1 = [3]
b = 0
retrainModel = False
retrainName = r'C:\Users\stoer\Imaging\results_new0001.hdf5'     
for lr in learningRate:
    VALlossWhileTraining = []
    VALaccuracyWhileTraining = []
    lossWhileTraining = []
    accuracyWhileTraining = []
    num = epoch1[b]
    print(lr)
    if retrainModel:
        model = keras.engine.saving.load_model(retrainName)
        model.compile(SGD(lr, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])
    else:
        model = get_model(lr)

    
    # get the data generators
    train_gen, val_gen = get_pcam_generators('/Users/s150055/Documents/TUe/Programming/Python/Imaging')
    
    
    # save the model and weights
    model_name = 'NewModel'
    model_filepath = model_name + '.json'
    weights_filepath = model_name + '_weights.hdf5'
    
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
    
    history = model.fit_generator(train_gen, steps_per_epoch=train_steps,
                        validation_data=val_gen,
                        validation_steps=val_steps,
                        epochs=num,
                        callbacks=callbacks_list)
    
    # Get data 
    VALlossWhileTraining.append(history.history.get('val_loss'))
    lossWhileTraining.append(history.history.get('loss'))
    VALaccuracyWhileTraining.append(history.history.get('val_acc'))
    accuracyWhileTraining.append(history.history.get('acc'))
    ExportData(VALlossWhileTraining, VALaccuracyWhileTraining, lossWhileTraining, accuracyWhileTraining, lr)
    f1 = r"C:\\Users\\stoer\\Imaging\\results_new" + str(lr)[2:] + ".hdf5"
    keras.engine.saving.save_model(model, f1, overwrite=True, include_optimizer=True)
    b += 1