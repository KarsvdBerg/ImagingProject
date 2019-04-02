## Exercise 1
The ROC curve is the *Receiver Operating Characteristic* curve and is used to evaluate the quality of the classifier output.
When measuring the performance of deep learning we use *Area Under the Curve (AUC)* as a measurement value for the ROC curve. A larger area, and thus a higher AUC, under the curve the better. The curve has the false positive rates on the x-axis and the true positives on the y-axis. [1][2] For the ROC curve an additional validation data set has to be loaded in to the model. This is done by adding the following lines of code to the get_pcam_generators function:
```
all_val = datagen.flow_from_directory(valid_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=16000,
                                             class_mode='binary')
     
     
        
     return train_gen, val_gen, all_val
```
After the model is trained de predictions can be visualized in the ROC curve with the following lines of code:
```
val_images = all_val[0][0]
val_labels = all_val[0][1]

#Testing our validation data
predictions = model.predict([val_images])

#Generate ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, _ = roc_curve(val_labels, predictions)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', 
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
print('AUC: {:.2f}'.format(roc_auc))
```
As can be seen the program calculates the true and false positives of predictions and plots them against each other to form the ROC curve. After that it gives us the AUC score by calulating the area under the curve. 

## Exercise 2
Convolution layers can detect paterns in images, with deeper layers being able to "detect" more complicated shapes than the first few layers

Result running original code:
```
Found 16000 images belonging to 2 classes.

Epoch 1/3
4500/4500 [==============================] - 308s 68ms/step - loss: 0.4637 - acc: 0.7831 - val_loss: 0.4312 - val_acc: 0.7996
Epoch 00001: val_loss improved from inf to 0.43117

Epoch 2/3
4500/4500 [==============================] - 300s 67ms/step - loss: 0.3930 - acc: 0.8255 - val_loss: 0.3619 - val_acc: 0.8418117/4500 [==========================>...] - ETA: 23s - loss: 0.3949 - acc: 0.8246
Epoch 00002: val_loss improved from 0.43117 to 0.36187

Epoch 3/3
4500/4500 [==============================] - 307s 68ms/step - loss: 0.3434 - acc: 0.8512 - val_loss: 0.3469 - val_acc: 0.8501661/4500 [=======================>......] - ETA: 51s - loss: 0.3464 - acc: 0.8498 - ETA: 36s - loss: 0.3458 - acc: 0.8503
Epoch 00003: val_loss improved from 0.36187 to 0.34690,
```
Now with the following changes in te code:
```
model = Sequential()

     model.add(Conv2D(first_filters, kernel_size, activation = 'relu', padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
     model.add(MaxPool2D(pool_size = pool_size))

     model.add(Conv2D(second_filters, kernel_size, activation = 'relu', padding = 'same'))
     model.add(MaxPool2D(pool_size = pool_size))

     model.add(Conv2D(second_filters, kernel_size, activation = 'relu', padding = 'same'))
     model.add(MaxPool2D(pool_size = pool_size))
     
     model.add(Flatten())
     model.add(Dense(1, activation = 'sigmoid'))
```
This results in :
```
Epoch 1/3
4500/4500 [==============================] - 314s 70ms/step - loss: 0.4840 - acc: 0.7702 - val_loss: 0.4164 - val_acc: 0.8158
Epoch 00001: val_loss improved from inf to 0.41643

Epoch 2/3
4500/4500 [==============================] - 321s 71ms/step - loss: 0.3945 - acc: 0.8247 - val_loss: 0.3440 - val_acc: 0.8518 - ETA: 4:31 - loss: 0.4197 - acc: 0.8068
Epoch 00002: val_loss improved from 0.41643 to 0.34402

Epoch 3/3
4500/4500 [==============================] - 319s 71ms/step - loss: 0.3538 - acc: 0.8462 - val_loss: 0.3199 - val_acc: 0.8626 - ETA: 2:30 - loss: 0.3606 - acc: 0.8421
Epoch 00003: val_loss improved from 0.34402 to 0.31994
```

## Exercise 3

[1] https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html \
[2] https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5
