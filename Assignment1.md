# Assignment 1
## Exercise 1
The presence of metastases is an important component of breast cancer staging [1]. Pathologist evaluate this by examination of histological preparations of lymph nodes. Currently, the lymph nodes are assessed manually. This is a rather complicated method that can result in a missed metastases. Consequently, this will result in an inadequate treatment for the patient. Introducing whole-slide images, will optimize the assessment of metastases [2]. It will also make the detection of metastases more accurate and objective. 
## Exercise 2
The current role of the pathologist is reviewing the biological tissue. It is a labor intensive and error-prone process [3]. By introducing whole-slide imaging, the workflow in a pathology lab will change. The detection of the metastases by deep learning will increase the speed and accuracy of the detection. It will save the pathologists a lot of time, they will now have a more evaluating role, instead of a determining role in the detection of metastases.
## Exercise 3
A neural network that works with small images for registration is Fully Convolutional Network. This is an neural network that consists of only convolutional layers. It consists of layers that produce outputs for arbitrary input sizes, this enables the model to be trained on small images and then applied to larger images such as whole slide images. [1] In the CAMELYON16 competition this model was used in 2 different methods from two different research groups, HTW-Berlin and CULab III. CULab III obtained the fourth highest FROC score (0.703) of all research teams participating in the competition.  Fully Convolutional NetworkShelhamer et al. explains how fully conventional networks (FNW) operate and where they originate from. Each layer output in a FNW is a convolutional network with a 3D array of size h x w x d, where h and w are spatial dimensions, and d is the feature dimension. The first layer is the image, with pixel size h x w and d features. Location in the higher layers correspond to the locations in the image they are connected to, which are called receptive field. A real-valued loss function composed with a FCN defines a task. The stochastic gradient decent computed on whole images will be the same as the stochastic gradient decent on the receptive fields, which are used as minibatches. When these receptive fields overlap significantly, both feedforward computation and backpropagation are much more efficient when computed layer-by-layer over an entire image.[2]
## Exercise 4
CODE:
# assignment 1.4 opening and showing images


"first import libaries"

import matplotlib.pyplot as plt

#import numpy as np
import imageio as imageio
#import pathlib as path

"uploading/reading images"
zero_1 = imageio.imread('C:\\Users\\s149611\\OneDrive - TU Eindhoven\\Project imaging\\dataaa\\train\\0_hoi\\fe0d6772ada1ceb0d662c586606a4b1549c63c85.jpg')
zero_2 = imageio.imread('C:\\Users\\s149611\\OneDrive - TU Eindhoven\\Project imaging\\dataaa\\train\\0_hoi\\hoi.jpg')
zero_3 = imageio.imread('C:\\Users\\s149611\\OneDrive - TU Eindhoven\\Project imaging\\dataaa\\train\\0_hoi\\ee1e3a37906aa77883904b624446b456561c3f55.jpg')
zero_4 = imageio.imread('C:\\Users\\s149611\\OneDrive - TU Eindhoven\\Project imaging\\dataaa\\train\\0_hoi\\dfdd463eb3e0b9992a35df3d63ef1d36a93cb21d.jpg')

one_1 = imageio.imread('C:\\Users\\s149611\\OneDrive - TU Eindhoven\\Project imaging\\dataaa\\train\\1\\000af35befdd9ab2e24fac80fb6508dfd1edd172.jpg')
one_2 = imageio.imread('C:\\Users\\s149611\\OneDrive - TU Eindhoven\\Project imaging\\dataaa\\train\\1\\000d3de1f31201b54cf82572c10099606f33c791.jpg')
one_3 = imageio.imread('C:\\Users\\s149611\\OneDrive - TU Eindhoven\\Project imaging\\dataaa\\train\\1\\00a2a1175108c1c63970e01b71e664cccc10e5ec.jpg')
one_4 = imageio.imread('C:\\Users\\s149611\\OneDrive - TU Eindhoven\\Project imaging\\dataaa\\train\\1\\00a68ce6c1b6114f87823136e1b10b487b9358f1.jpg')

plt.imshow(zero_1)
plt.show()
plt.imshow(zero_2)
plt.show()
plt.imshow(zero_3)
plt.show()
plt.imshow(zero_4)
plt.show()


plt.imshow(one_1)
plt.show()
plt.imshow(one_2)
plt.show()
plt.imshow(one_3)
plt.show()
plt.imshow(one_4)
plt.show()

References

[1] Ehteshami Bejnordi, B., Veta, M., Johannes van Diest, P., van Ginneken, B., Karssemeijer, N., Litjens, G., ... Venâncio, R. (2017). Diagnostic Assessment of Deep Learning Algorithms for Detection of Lymph Node Metastases in Women With Breast Cancer. JAMA, 318(22), 2199–2210. https://doi.org/10.1001/jama.2017.14585

[2] Shelhamer, E., Long, J., & Darrell, T. (2017). Fully Convolutional Networks for Semantic Segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 39(4), 640–651. https://doi.org/10.1109/TPAMI.2016.2572683
