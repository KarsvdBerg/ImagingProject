Method

During this project a convolutional neural network (CNN) is made. CNN is a kind of network architecture and mostly used for image recognition. It has convolutions inside, which can see the edges of an object. In this case, the network is trained to recognize the edges of metastases, 
 Keras

###Model: 
The model type that we will be using is Sequential. Sequential is the easiest way to build a model in Keras, because it allows to build a model layer by layer.
In this model different kinds of layers are used, namely Conv2D, flatten, dense and maxpooling layers. 
Conv2D layers are convolutional layers that will deal with the input images, which are seen as 2D matrices.
The connection between the convolution and dense layers will be done by the flatten layer.
In model.add(Dense ..... activation = ...) we have an activation function. In our model we have used ReLU and Sigmoid. Instead of using Sigmoid, Softmax can also be used. Softmax ensures that the output can be interpreted as probabilities by suming up the output up to 1. The Dense is a standard layer type. It will be used for our output layer.

The ‘add()’ function makes it possible to add layers to the model. 
###Compiling
To compile the model three parameters are taken, namely an optimizer, a loss and a metrics. At the moment are code consist of the following line: model.compile(SGD(lr=0.01, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy']).
The optimizer controls the learning rate. There are different kinds of methods but we used the Stochastic gradient descent (SGD). The learning rate measures how fast the optimal weights are calculated. A smaller learning rate can lead to more precise weights, but it will take longer to determine them. [6] With the loss function it can be evaluated how good the predicted probabilities are. It should return low values for good predictions. For the loss function 'binary_crossentropy' is applied. The performance of a classification model whose output is a probability value between 0 and 1 can be measured by cross-entropy loss. In this case it is binary, because a binary classifier is trained. [7] To see the accuracy score on the validation set when training the model, the 'accuracy' metric is used.
Getting data generators
Save the model and weights
Checkpoint, tensorboard callbacks
Next, the model will be trained by model.fit(). Three epochs are used, which means that the model will cycle three times trough the data. The model will improve up to a certain point if you run more epochs. After that certain point, the model will stop improving during each epoch. [6] 
The ROC curve is used to see the actual predictions the model has made. The ROC curve is the 'Receiver Operating Characteristic' curve and is used to evaluate the quality of the classifier output. The curve has the false positive rates on the x-axis and the true positives on the y-axis. A larger area under the curve is usually better. [8]
## References
[6] https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5   </br>
[7] https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html </br>
[8] https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
