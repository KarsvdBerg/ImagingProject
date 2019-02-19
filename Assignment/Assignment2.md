# Assignment 2
## Exercise 1

| Layers | Neurons per layer | Loss | Accuracy |
| --- | --- | --- | --- |
| 1   | 64| 0.18693242855668069 |0.9476 |
| 2|64| 0.188774238589406 | 0.9443 |
| 5| 64| 0.1126803500131704  | 0.9655 |
| 8 | 64 | 0.11755173907494172 |0.9654 |
|12 |64|0.12003407826758922 |0.9673|
| 15|64|0.15983885976038872|0.9569|
| 1   | 1| 1.6259437475204468 | 0.3532 |
| 1| 128| 0.1727910432316363 | 0.9507 |
| 1 | 192 | 0.1618253667011857|0.9533|
|1|256| 0.16680156881958247|0.9499|

Using a different amount of layers with 64 neurons in 1 layer

| Number of layers | Loss test | Accuracy test | Loss train  | Accuracy train |
| --- | --- | --- | --- | --- |
| 1 |	0.18280 |	0.9474	| 0.178546 |	0.94925 |
| 2 |	0.14402	| 0.9578	| 0.13013 |	0.96390 |
| 3 |	0.12360	| 0.9607	| 0.101313 |	0.96972 |
| 4 |	0.12250	| 0.9635  |	0.09265 |	0.97348 |
| 5 |	0.11026	| 0.9663	| 0.07358 |	0.97794 |
| 6 |	0.11596	| 0.9666	| 0.07348 |	0.97824 |
| 7 |	0.11260	| 0.965	| 0.066571 |	0.97970 |
| 8 |	0.13257	| 0.9638	| 0.084182 |	0.97418 |
| 9 |	0.13278	| 0.9579	| 0.079226 |	0.975 |
| 10 |	0.11001 |	0.9674	| 0.058272	|0.98288 |

![alt text](https://github.com/KarsvdBerg/ImagingProject/blob/master/Assignment2.1A.png "Logo Title Text 1")
![alt text](https://github.com/KarsvdBerg/ImagingProject/blob/master/Assignment2.1B.png "Logo Title Text 1")

With more layers, the accuracy generally goes up and the loss generally goes down. To see if the found differences are relevant or if the standart deviation is around the same value. 

|Number of measurement 	|Loss test	|Accuracy test	|Loss train	|Accuracy train|
| --- | --- | --- | --- | --- |
|1	|0.12788309332542122	|0.9609|	0.11190716571885126	|0.9679629629629629|
|2	|0.12318471043929458	|0.963|	0.10236615612440639	|0.9704074074074074|
|3	|0.12142142855897546	|0.9645|	0.10040908840409031	|0.9708888888888889|
|4	|0.12487356092911214	|0.9612| 0.10026886706495726	|0.9707222222222223|
|5	|0.12292111209668219	|0.9609|	0.10251085377815697	|0.9705370370370371|
|Standard deviation|	0.002465|	0.00160156	|0.00482022	|0.00121056|

Now the distribution of neurons in 3 layers is changes and the total number of neurons is kept at 192

|Number of neurons layer 1|	Number of neurons layer 2|	Number of neurons layer 3	|Loss test	|Accuracy test	|Loss train	|Accuracy train|
| --- | --- | --- | --- | --- | --- | --- |
|64	|64	|64	|0.12788	|0.9609	|0.11190	|0.96796|
|96	|64	|32	|0.11242	|0.965	|0.085901	|0.97531|
|32	|64	|96	|0.15864	|0.9497	|0.139137	|0.95738|
|96	|48	|48	|0.11561	|0.9649	|0.090108	|0.97435|
|48	|96	|48	|0.11826	|0.9655	|0.099019	|0.97081|
|48	|48	|96	|0.14430	|0.9559	|0.122424	|0.96338|
|110	|41	|41	|0.11108	|0.9677	|0.085880	|0.97546|
|132	|30	|30	|0.11218	|0.9668	|0.084590	|0.97605|


## Exercise 2
1.	loss: 0.3237 - acc: 0.9104 - val_loss: 0.3462 - val_acc: 0.9023
2.	loss: 0.1105 - acc: 0.9674 - val_loss: 0.1362 - val_acc: 0.9610
3.	loss: 0.2696 - acc: 0.9248 - val_loss: 0.3037 - val_acc: 0.9135



## Exercise 3
### Code
```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.callbacks import TensorBoard

# load the dataset using the builtin Keras method
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```
Add class names for extra visualisation on the data, define the four categories that will be assigned to the
adequate number.
```
class_names = ['loopy digits','vertical digits','curly digits','other','other',
               'curly digits','loopy digits','vertical digits','loopy digits','loopy digits']

print('Dimensionality of the training image dataset and labels:')
print(train_images.shape)
print(train_labels.shape)

print('Dimensionality of the test image dataset and labels:')
print(test_images.shape)
print(test_labels.shape)

# show the first image in the dataset
plt.figure()
plt.imshow(train_images[0], cmap='gray_r', vmin=0, vmax=255)
plt.title('First image in the dataset')
```
Adding class_names to the for loop to create a 4x4 grid for a little bit extra visualisation 
```
# show the first 16 images in the dataset in a 4x4 gird
fig = plt.figure(figsize=(10,10))
for n in range(16):
    ax = fig.add_subplot(4, 4, n + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[n], cmap='gray_r', vmin=0, vmax=255)   
    plt.xlabel(class_names[train_labels[n]])    # Display the digits with their assigned class names 
    #plt.axis('off')
fig.suptitle('First 16 images in the dataset')
plt.show()

# print the labels of the first 16 images in the dataset
print('Labels of the first 16 images in the dataset:')
print(train_labels[:16])


# derive a validation set from the training set
# the original training set is split into 
# new training set (90%) and a validation set (10%)
train_images, val_images = train_test_split(train_images, test_size=0.10, random_state=101)
train_labels, val_labels = train_test_split(train_labels, test_size=0.10, random_state=101)

print('Dimensionality of the new training image dataset and labels:')
print(train_images.shape)
print(train_labels.shape)

print('Dimensionality of the validation image dataset and labels:')
print(val_images.shape)
print(val_labels.shape)

def plt_classes(y, num_class=10):
    plt.figure()
    plt.hist(y, bins=range(0,num_class+1), align='left', rwidth=0.9)
    plt.xlabel('Class')
    plt.ylabel('Class count')
    plt.xticks(range(0,num_class))
    plt.title('Class distribution')
    
# show the class label distribution in the training dataset
plt_classes(train_labels)
# show the class label distribution in the validation dataset
plt_classes(val_labels)

# the shape of the data matrix is NxHxW, where
# N is the number of images,
# H and W are the height and width of the images
# keras expect the data to have shape NxHxWxH, where
# C is the channel dimension
train_images = np.reshape(train_images, (-1,28,28,1)) 
val_images = np.reshape(val_images, (-1,28,28,1))
test_images = np.reshape(test_images, (-1,28,28,1))

# convert the datatype to float32
train_images = train_images.astype('float32')
val_images = val_images.astype('float32')
test_images = test_images.astype('float32')

# normalize our data values to the range [0,1]
train_images /= 255
val_images /= 255
test_images /= 255

#class label preprocessing for keras

# we see that we have a 1D-array with length 54000
print(train_labels.shape) 
# since we have 10 different classes, what does this array look like?
# let's look at the first 20 labels
print(train_labels[:20]) 
```
Assign each number to their adequate group, so: 1 and 7 = 0 (vertical digits), 0,6,8 and 9 = 1 (loopy digits)
2 and 5 = 2 (curly digits) and 3 and 4 = 3 (other) 
```
# Convert the new categorical values back to integers 
# Vertical digits = 0, Loopy digits = 1, Curly digits = 2 and Other = 3
for n in range(len(train_labels)):
    if train_labels[n] == 1 or train_labels[n] == 7:
        train_labels[n] = 0
    elif (train_labels[n] == 0 or train_labels[n] == 6 or 
              train_labels[n] == 8 or train_labels[n] == 9):
        train_labels[n] = 1     
    elif train_labels[n] == 2 or train_labels[n] == 5:
        train_labels[n] = 2
    elif train_labels[n] == 3 or train_labels[n] == 4:
        train_labels[n] = 3

for n in range(len(val_labels)):
    if val_labels[n] == 1 or val_labels[n] == 7:
        val_labels[n] = 0
    elif (val_labels[n] == 0 or val_labels[n] == 6 or 
              val_labels[n] == 8 or val_labels[n] == 9):
        val_labels[n] = 1     
    elif val_labels[n] == 2 or val_labels[n] == 5:
        val_labels[n] = 2
    elif val_labels[n] == 3 or val_labels[n] == 4:
        val_labels[n] = 3

for n in range(len(test_labels)):
    if test_labels[n] == 1 or test_labels[n] == 7:
        test_labels[n] = 0
    elif (test_labels[n] == 0 or test_labels[n] == 6 or 
              test_labels[n] == 8 or test_labels[n] == 9):
        test_labels[n] = 1     
    elif test_labels[n] == 2 or test_labels[n] == 5:
        test_labels[n] = 2
    elif test_labels[n] == 3 or test_labels[n] == 4:
        test_labels[n] = 3

# Show the first 10 training labels with corresponding categorical labels 
labels = []      
categories = ['vertical digits', 'loopy digits', 'curly digits', 'other']
for n in range(len(train_labels)):
    labels.append(categories[train_labels[n]])
print("First 10 training labels with corresponing categorical label")
for t, l in zip(train_labels[:10], labels[:10]):
    print("{} = {}".format(t,l))


# convert 1D class arrays to 4D class matrices
train_labels = np_utils.to_categorical(train_labels, 4)
val_labels = np_utils.to_categorical(val_labels, 4)
test_labels = np_utils.to_categorical(test_labels, 4)


# check the output
print(train_labels.shape)
# this is now a one-hot encoded matrix
print(train_labels[:20])

#Build the model
model = Sequential()
# flatten the 28x28x1 pixel input images to a row of pixels (a 1D-array)
model.add(Flatten(input_shape=(28,28,1))) 
# fully connected layer with 64 neurons and ReLU nonlinearity
model.add(Dense(64, activation='relu'))
# output layer with 4 nodes (one for each class) and softmax nonlinearity
model.add(Dense(4, activation='softmax'))


# compile the model
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# use this variable to name your model
model_name="my_first_model"

# create a way to monitor our model in Tensorboard
tensorboard = TensorBoard("logs/{}".format(model_name))

# train the model
model.fit(train_images, train_labels, batch_size=32, epochs=10, verbose=1, validation_data=(val_images, val_labels), callbacks=[tensorboard])
score = model.evaluate(test_images, test_labels, verbose=0)


print("Loss: ",score[0])
print("Accuracy: ",score[1])
```
Visualize and predict the outcome of the new model. The output can be seen in the table below this block of code
```
# make prediction 
predictions = model.predict(test_images)

# Show the predicted label, the actual test label and the categorical label that is predicted 
prediction_values = []
for n in range(len(predictions)):
    prediction_values.append(np.argmax(predictions[n]))

prediction_labels = []
for n in range(len(predictions)):
    prediction_labels.append(labels[prediction_values[n]])

print("left prediction, middle test labels and right label names")
for p, t, l in zip(prediction_values[:20], test_labels[:20], prediction_labels):
    print("{} = {} = {}".format(p,t,l))
```

### Output
| Prediction label | Test labels | Categorical prediction label |
| - | -------| -------------|
| 0 | [1. 0. 0. 0.] | vertical digits |
| 2 | [0. 0. 1. 0.] | loopy digits |
| 0 | [1. 0. 0. 0.] | vertical digits |
| 1 | [0. 1. 0. 0.] | other |
| 3 | [0. 0. 0. 1.] | curly digits |
| 0 | [1. 0. 0. 0.] | vertical digits |
| 3 | [0. 0. 0. 1.] | curly digits |
| 1 | [0. 1. 0. 0.] | other |
| 1 | [0. 0. 1. 0.] | other |
| 1 | [0. 1. 0. 0.] | other |
| 1 | [0. 1. 0. 0.] | other |
| 1 | [0. 1. 0. 0.] | other |
| 1 | [0. 1. 0. 0.] | other | 
| 1 | [0. 1. 0. 0.] | other |
| 0 | [1. 0. 0. 0.] | vertical digits |
| 2 | [0. 0. 1. 0.] | loopy digits |
| 1 | [0. 1. 0. 0.] | other |
| 0 | [1. 0. 0. 0.] | vertical digits |
| 3 | [0. 0. 0. 1.] | curly digits |
|3 | [0. 0. 0. 1.] | curly digits

