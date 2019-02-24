# Assignment 4
## Exercise 1
Transfer learning is a method in machine learning. Transfer learning is different than other, more traditional machine learning. Transfer learning makes use of already excisting source tasks in a related target task [1]. Transfer learning makes sense when:
  1. Task A and B have the same input
  2. Task A has a lot more data than task B
  3. Low level features from A can be helpful for B

It makes sense to apply transfer learning from ImageNet to the Patch-CAMELYON dataset, as the datasets have the same input, eg. images. Both datasets can be used in image recognition, so the implementation of both datasets is, in a way, the same. 

## Exercise 2
When changing the weights argument to *None* instead of *'imagenet* (pre-training on ImageNet) the validation-loss and -accuracy of the dataset is affected. When training with the weights set to *'imagenet* the validation loss declines until the second last epoch were it peaks at  **val_loss = 0.1710** with an accuracy of **val_acc = 0.9337**. See output below:

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Epoch 9**
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Epoch 00008: val_loss improved from 0.28843 to 0.27055, saving model to    my_first_transfer_model_weights.hdf5
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Epoch 9/10
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;225/225 [==============================] - 46s 206ms/step - loss: 0.2264 - acc: 0.9146 - val_loss: 0.1710 - val_acc: 0.9337

When training with the weights set to *None* the validation-loss and -accuracy is negetively affected. The improvement of the validation-loss seems random as it does not improve form epoch 2 to 3, but does improve from epoch 4 to 5. It peaks at epoch 9 where the validation-loss is **val_loss = 0.518** with an accuracy of **val_acc = 0.7925**. See output below

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Epoch 2-3**
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Epoch 00001: val_loss improved from inf to 1.18490, saving model to my_first_transfer_model_weights.hdf5
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Epoch 2/10
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;225/225 [==============================] - 47s 207ms/step - loss: 0.5402 - acc: 0.7446 - val_loss: 3.1652 - val_acc: 0.5413

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Epoch 00002: val_loss did not improve from 1.18490
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Epoch 3/10
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;225/225 [==============================] - 47s 208ms/step - loss: 0.5032 - acc: 0.7697 - val_loss: 0.9143 - val_acc: 0.7137

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Epoch 4-5**
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Epoch 00003: val_loss improved from 1.18490 to 0.91433, saving model to my_first_transfer_model_weights.hdf5
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Epoch 4/10
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;225/225 [==============================] - 47s 207ms/step - loss: 0.4733 - acc: 0.7836 - val_loss: 0.8087 - val_acc: 0.7887

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Epoch 00004: val_loss improved from 0.91433 to 0.80866, saving model to my_first_transfer_model_weights.hdf5
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Epoch 5/10
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;225/225 [==============================] - 47s 207ms/step - loss: 0.4680 - acc: 0.7842 - val_loss: 1.0119 - val_acc: 0.7625

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Epoch 9**
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Epoch 00008: val_loss did not improve from 0.54272
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Epoch 9/10
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;225/225 [==============================] - 47s 209ms/step - loss: 0.4098 - acc: 0.8212 - val_loss: 0.5158 - val_acc: 0.8075

## Exercise 3


## References

[1] Torrey, L., Shavlik, J. (2009). Transfer Learning. University of Wisconsin
