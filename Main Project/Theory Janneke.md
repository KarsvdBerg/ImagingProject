
paper: https://link.springer.com/chapter/10.1007/978-3-030-00214-5_150

- relu linear activation function is most frequently used

paper: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.124.1000&rep=rep1&type=pdf

- Overfitting happens if the number of hidden neurons is too big.
- Sigmoidal acitvation zorgt voor faster learning
- they use fletcher-Reeves conjugate gradient
- the learning rate plays major role in NN performances, choosing a wrong value can conclude in unstability of the algorithm

paper: https://ieeexplore.ieee.org/abstract/document/8448814

- Since the image of melanoma cancer has no distinct feature, therefore, deep layer CNN cannot perform well for melanoma cancer detection due to overfitting problem. 
This problem arises when the model is trained too well. Consequently, it starts to have deleterious effect on the results. 
It is suggested that CNN architecture is more suitable for identifying texture-based images and it can avoid overfitting problems.
- we have used single convolution layer since there are few features to be learned, hence it can reduce the complexity of the CNN and avoid overfitting problem. 
- after that they used 3 fully connected layers

paper: https://link.springer.com/chapter/10.1007/978-3-319-60964-5_23
- uses a pre trained CNN architecture , because is saves time and money and it handy if you don't have a large database
- they use AlexNet, pretrained CNN architectures with architecture:
 Data, conv1, max pooling 1, conv2, max pooling2, conv3, conv4, conv5, max pooling 3, fully connected 1, fully connected 2, fully connected 3

paper: Stanitsas, P.; Cherian, A.; Truskinovsky, A. Active convolutional neural networks for cancerous tissue
recognition. In Proceedings of the International Conference on Image Processing (ICIP), Beijing, China,
17–20 September 2017.

-use multistage training scheme to overcome the overfitting problem

paper: https://www.mdpi.com/2076-3417/9/3/427

- max pooling prevents overfitting 

paper: https://ieeexplore.ieee.org/abstract/document/8512386

  - data augmentation can be used to increase model complexity and reduce overfitting
  - unsing rectified linear unit (ReLu) can reduce training time
  - dropout layer can be added to reduce overfitting in imbalance dataset
  - smaller filter size can be used to retain more pixel information
  - back to back convolution layer with padding can be used to maintain more pixel information in shrinking spatial information but increase model layer depth
  - Adding more than two max pool layers with stride 2 will also increase objective loss resulting low classifier accuracy
  - had 3 classes (cancer cells)
  - they use 8 convulation layers, 3 max pooling layers (c,c,p,c,c,p,c,c,c,p,c) and one fully connected layer
  
paper: 	https://ieeexplore.ieee.org/abstract/document/7727519 2016

  - good general explanation deep learning with image classification of breast cancer
  - "general CNN architecture Input - conv+pooling layer(s) - fully connected layer(s) - output
  - pooling layers -> reduces the spatial size of the representation -> reduces the numbe rof parameters and computations required by the network -> helps in overfitting control
  - fully connected layer -> 
  
paper:  L.G. Hafemann, L.S. Oliveira, P. Cavalin, " Forest species recognition using deep convolutional neural networks", International conference of pattern recognition, pp. 1103-1107, 2014
 
 -best results were achieved by reducing the dimensionality of the images, in this work the original 700×460 images were reduced to 350×230, resampling using pixel area relation. This can prevent the model from overfitting the training set.

paper: https://www.sciencedirect.com/science/article/pii/S0925231216001004

  - two alterning convolutional layer,max pooling layers, two full connection layers and a final classifitation layer
  
paper : https://ieeexplore.ieee.org/abstract/document/7312934 2016
  - Histological tissue images can be characterized by two types of approaches. The first one is based on explicit segmentation to extract structure properties, such as nuclei shape, glandular unit shape, etc., while the second one is a global approach based on texture representation. Since segmentation of histological tissue images is not a trivial task and can be prone to errors, we have chosen a global approach based on state-of-the-art texture representation.
  
paper: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0177544
  - input - convo +max pooling - convo +max pooling - convo +max pooling - convo +max pooling - convo +max pooling - fully connected- fully connected - output layer
  
paper: https://www.sciencedirect.com/science/article/pii/S0031320317302005
  - same "standard" structure 

