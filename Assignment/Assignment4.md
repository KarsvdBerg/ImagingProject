# Assignment 4
## Exercise 1
Transfer learning is a method in machine learning. Transfer learning is different than other, more traditional machine learning. Transfer learning makes use of already excisting source tasks in a related target task [1]. Transfer learning makes sense when:
  1. Task A and B have the same input
  2. Task A has a lot more data than task B
  3. Low level features from A can be helpful for B

It makes sense to apply transfer learning from ImageNet to the Patch-CAMELYON dataset, as the datasets have the same input, eg. images. Both datasets can be used in image recognition, so the implementation of both datasets is, in a way, the same. 

## Exercise 2
When changing the weights argument to *None* instead of *'imagenet* (pre-training on ImageNet) the validation-loss and -accuracy of the dataset is affected. When training with the weights set to *'imagenet* the validation loss declines until the second last epoch were it peaks at  **val_loss = 0.1710** with an accuracy of **val_acc = 0.9337**. See output below:

```
  **Epoch 9**
  Epoch 00008: val_loss improved from 0.28843 to 0.27055, saving model to    my_first_transfer_model_weights.hdf5
  Epoch 9/10
  225/225 [==============================] - 46s 206ms/step - loss: 0.2264 - acc: 0.9146 - val_loss: 0.1710 - val_acc: 0.9337
```

When training with the weights set to *None* the validation-loss and -accuracy is negetively affected. The improvement of the validation-loss seems random as it does not improve form epoch 2 to 3, but does improve from epoch 4 to 5. It peaks at epoch 9 where the validation-loss is **val_loss = 0.518** with an accuracy of **val_acc = 0.7925**. See output below
 
 ```
  **Epoch 2-3**
  Epoch 00001: val_loss improved from inf to 1.18490, saving model to my_first_transfer_model_weights.hdf5
  Epoch 2/10
  225/225 [==============================] - 47s 207ms/step - loss: 0.5402 - acc: 0.7446 - val_loss: 3.1652 - val_acc: 0.5413

  Epoch 00002: val_loss did not improve from 1.18490
  Epoch 3/10
  225/225 [==============================] - 47s 208ms/step - loss: 0.5032 - acc: 0.7697 - val_loss: 0.9143 - val_acc: 0.7137

  **Epoch 4-5**
  Epoch 00003: val_loss improved from 1.18490 to 0.91433, saving model to my_first_transfer_model_weights.hdf5
  Epoch 4/10
  225/225 [==============================] - 47s 207ms/step - loss: 0.4733 - acc: 0.7836 - val_loss: 0.8087 - val_acc: 0.7887

  Epoch 00004: val_loss improved from 0.91433 to 0.80866, saving model to my_first_transfer_model_weights.hdf5
  Epoch 5/10
  225/225 [==============================] - 47s 207ms/step - loss: 0.4680 - acc: 0.7842 - val_loss: 1.0119 - val_acc: 0.7625

  **Epoch 9**
  Epoch 00008: val_loss did not improve from 0.54272
  Epoch 9/10
  225/225 [==============================] - 47s 209ms/step - loss: 0.4098 - acc: 0.8212 - val_loss: 0.5158 - val_acc: 0.8075
```
## Exercise 3
Dropout is a regulation method that prevents neural networks from overfitting [2]. In this technique randomly selects neurons that are ignored during training. By doing this you create a neural network that won't be to specialized to the training data that is used. When leaving certain neurons out of the training other neighbouring neurons have to step in. The effect is that the network  becomes less sensitive to the specific weights of neurons.[3] Removing the dropout layer in the neural network causes an increase in the loss, and thus a worse model. The validation loss has an optimal and final value of **val_loss = 0.6537** with a accuracy of **val_acc = 0.7538**. The training loss has an optimal and final value of **loss = 0.4009** with an accuracy of **acc = 0.8208**. The results show that 
**loss > val_loss** which could be linked to overfitting.

## References

[1] Torrey, L., Shavlik, J. (2009). Transfer Learning. University of Wisconsin
[2] Documentation python *help(Dropout)* 
[3] Dropout Regularization in Deep Learning Models With Keras. (n.d.). Retrieved February 24, 2019, from https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
