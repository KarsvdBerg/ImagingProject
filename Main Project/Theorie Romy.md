Our task: develop a deep neural network model that can identify metastases in image patches from histological preparations of lymph node tissue from breast cancer patients.

https://eu.udacity.com/course/deep-learning--ud730 --> this is a free course about machine learning and deep learning. Maybe it's interesting.

_ZSO 01-03_

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

_ZSO 13-03_

About batch normalization
https://gluon.mxnet.io/chapter04_convolutional-neural-networks/cnn-batch-norm-scratch.html 
https://towardsdatascience.com/dont-use-dropout-in-convolutional-networks-81486c823c16
