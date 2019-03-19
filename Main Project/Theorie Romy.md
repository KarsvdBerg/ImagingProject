Our task: develop a deep neural network model that can identify metastases in image patches from histological preparations of lymph node tissue from breast cancer patients.

https://eu.udacity.com/course/deep-learning--ud730 --> this is a free course about machine learning and deep learning. Maybe it's interesting.

## _ZSO 01-03_

Neural networks:
- Are a class of models within machine learning
- Are a specific set of algorithms
- 'Deep' in 'deep neural network' stand for the amount of layers
- Learn from obsercational data, figure out their own solution to the problem. 

4 kinds of network architectures:
- Unsupervised pre-trained networks
- Convolutional neural networks
- Recurrent neural networks --> used for example for speech recognition
- Recursive neural networks --> hierarchical network

Convolutional neural network (CNN) is mostly used for image recognition, so I think we need to use that network.
It has convolutions inside, which can see the edges of an object. In our case, we could train the network to recognize the edges of metasases.

There are multiple methods which we can use for a CNN.
- Back propagation --> compute the partial derivatives of a function. Compute the function gradient at each iteration.
- Stochastic gradient descent
- Learning rate decay --> the performance of the training can be increased and the training time can be reduced, when you adapt the learning rate for the stochastic gradient descent. The effect of the learning rate is that is can learn quickly good weights and fine tuning them later. The learning rate can be decreased based on the epoch. 
- Dropout. This technique is used for large networks that have a high risk of overfitting. In this method you randomly drop units from the neural network during training.
- Max Pooling. This is a discretization process. To help overfitting by providing an abstract form of the representation (done by for example downsampling). It is done by applying a filter to usually non-overlapping subregions of the initial representation.
- Batch normalization. Helps relaxing the deep networks a little, because they require careful tuning of weight initialization and learning parameters. We've done this in assignment 3 by using mini batches. By this process, higher learning rates can be used.
- Long short-term memory.
- Transfer learning. See Assignment 4.1

## _ZSO 13-03_

About batch normalization
https://gluon.mxnet.io/chapter04_convolutional-neural-networks/cnn-batch-norm-scratch.html 
https://towardsdatascience.com/dont-use-dropout-in-convolutional-networks-81486c823c16

## _ZSO 20-03_

Onderstaande puntjes zijn handig voor in de Methode. Bovendien kunnen sommige dingen misschien worden gebruikt om onze code te verduidelijken. 
* The model type that we will be using is Sequential. Sequential is the easiest way to build a model in Keras. It allows you to build a model layer by layer.
* We use the ‘add()’ function to add layers to our model.
* Conv2D layers are convolutional layers that will deal with the input images, which are seen as 2D matrices.
* The Kernel size is the size of the filter matrix for the convolution that we use.
* The connection between the convolution and dense layers will be done by the flatten layer.
* in model.add(Dense ..... activation = ...) we have an activation function. In our model we have used ReLU and Sigmoid. Instead of using Sigmoid, Softmax can also be used. Softmax ensures that the output can be interpreted as probabilities by suming up the output up to 1. The Dense is a standard layer type. It will be used for our output layer.

### Compiling the model
