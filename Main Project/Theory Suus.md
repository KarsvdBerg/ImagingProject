## Deep Learning

Deep learning has Artificial neural networks are common machine learning techniques, that simulate the mechanisms of learning [1]. Neural networks can contain more layers. Convolution neural network is a deep learning algorithm that takes an image as an input. There are various architectures of CNNs available which have been key in building algorithms which power and shall power AI as a whole in the foreseeable future. Some of them have been listed below[3][4]:

1. *LeNet-5*: 7-level convolutional network by Lecun et al. Recognises hand written number on checks. 

2. *AlexNet*:Combination of Max-Pooing, dropout and data augumentation, ReLu axtivations, SGD with momentum.Trained for 6 days (???)

3. *GGNet*: 16 convolutional layers. Very uniform architecture. 138 million parameters.

4. *GoogLeNet*: 22 layers with reduced parameters (4 million)

5. *ResNet*: Skips connections and features heavy batch normalization.




I found this website with several deep learning methods. Among them was transfer learning. 
1. Back-propagation
2. Stochatic gradient descent
3. Learning rate decay
Increase performance and reduce training time.
 4. Dropout
Very useful when having a lot of parameters. Units are dropped from the neural network during training. This prevents overfitting 'Dropout has been shown to improve the performance of neural networks on supervised learning tasks in vision, speech recognition, document classification and computational biology, obtaining state-of-the-art results on many benchmark datasets.'
5. Max pooling
Reducing dimensionality. 
6. Batch normalization

7. Long short-term memory
It has control on deciding when to let the input enter the neuron.
It has control on deciding when to remember what was computed in the previous time step.
It has control on deciding when to let the output pass on to the next time stamp.
10. Transfer learning

# Deep Learning Code
From this site I retrieved some lines of code that I think will work in our code, as in our code, the first layers are also added by model.add:https://www.datacamp.com/community/tutorials/convolutional-neural-networks-python

```fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(Dense(num_classes, activation='softmax')
```
The following code is a code I found on a site, this is based on the LeNet code, using Keras Retreved from the following site:https://medium.com/@mgazar/lenet-5-in-9-lines-of-code-using-keras-ac99294c8086.

```model = keras.Sequential()

model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32,32,1)))
model.add(layers.AveragePooling2D())

model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(layers.AveragePooling2D())

model.add(layers.Flatten())

model.add(layers.Dense(units=120, activation='relu'))

model.add(layers.Dense(units=84, activation='relu'))

model.add(layers.Dense(units=10, activation = 'softmax'))
```
I found a paper on histopathological cancer diagnosis. A deep learning technique, given by a code in the attachments of the paper, was compared to examination of a resident pathology. The CNN consists of 4 convolutional layers, 3 max-pooling layers and lastly 3 classification layers. See the paper for the full code. For the CNN they make use of the open-source 'deep-learning'libraries Theano 0.7 and pylearn 0.1. As it is impossible to put the whole slide images in the network at once, random small patches were extracted for training. Whole-slide results can consequently be obtained by applying the network to every pixel [5]. https://www.nature.com/articles/srep26286#rightslink 

https://cdn.jamanetwork.com/ama/content_public/journal/jama/936626/joi170113supp1_prod.pdf?Expires=2147483647&Signature=C3T3r5K2p0j1I0YUgFQVEmcjFVQJ-RCwaO46SqSesmqQcgYdzVnOhO3ubtApInIzAnDB-LUa0St7p0LjtE9GhQxX5eYtAC6fETgLgGn62tXFKJcXadhZ1P1UOGVds3hLBRsipn0X3g8lwULGA1sLg0csUKUP9ko0KoWO~1Tcwzb9RYu~V8Pur~7ijop5xhncqHfLZFeeI2Fyz9gzdUAAnlO-m0igw85JMuvTqngrdbjD3MtATfH-65f2oKIZEtRxcf56nRRaW5PxuqEl4QphxR30B2p2eYPOTFC8-CQPO1ZDg74CeT6ywGsxURWIWw686QQLPyrMmkUpz8R2HHH79w__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA This document showes all the deep learning methods of the CAMELYON16 challenge. Very useful, the parameters that they used are also stated.

# References
[1] https://books.google.nl/books?id=achqDwAAQBAJ&printsec=frontcover&dq=deep+learning&hl=en&sa=X&ved=0ahUKEwjoypnIhd7gAhVD7eAKHYueAj8Q6AEIOjAC#v=onepage&q=deep%20learning&f=false 

[2]https://medium.com/cracking-the-data-science-interview/the-10-deep-learning-methods-ai-practitioners-need-to-apply-885259f402c1 

[3] https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53

[4] https://medium.com/@sidereal/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5

[5]  Litjens, G. et al. Deep learning as a tool for increased accuracy and efficiency of histopathological diagnosis. Sci. Rep. 6, 26286; doi: 10.1038/srep26286 (2016).

https://arxiv.org/pdf/1409.1556.pdf --> Article on effect of convolutional network depth on accuracy in large scale image recognition. 'Our main contribution is a thorough evaluation of networks of increasing depth using an architecture with very small (3×3) convolution filters, which shows that a significant improvementon the prior-art configurations can be achieved by pushing the depth to 16–19
weight layers.' So a 16-19 weight layers optimal for image recognition of a large scale.



