
paper: https://link.springer.com/chapter/10.1007/978-3-030-00214-5_150

- relu linear activation function is most frequently used

paper: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.124.1000&rep=rep1&type=pdf

-Overfitting happens if the number of hidden neurons is too big.
-Sigmoidal acitvation zorgt voor faster learning
-they use fletcher-Reeves conjugate gradient
- the learning rate plays major role in NN performances, choosing a wrong value can conclude in unstability of the algorithm

paper: https://ieeexplore.ieee.org/abstract/document/8448814

- Since the image of melanoma cancer has no distinct feature, therefore, deep layer CNN cannot perform well for melanoma cancer detection due to overfitting problem. 
This problem arises when the model is trained too well. Consequently, it starts to have deleterious effect on the results. 
It is suggested that CNN architecture is more suitable for identifying texture-based images and it can avoid overfitting problems.
-we have used single convolution layer since there are few features to be learned, hence it can reduce the complexity of the CNN and avoid overfitting problem. 
-after that they used 3 fully connected layers

paper: https://link.springer.com/chapter/10.1007/978-3-319-60964-5_23
-uses a pre trained CNN architecture , because is saves time and money and it handy if you don't have a large database
- they use AlexNet, pretrained CNN architectures with architecture:
 Data, conv1, max pooling 1, conv2, max pooling2, conv3, conv4, conv5, max pooling 3, fully connected 1, fully connected 2, fully connected 3

paper: Stanitsas, P.; Cherian, A.; Truskinovsky, A. Active convolutional neural networks for cancerous tissue
recognition. In Proceedings of the International Conference on Image Processing (ICIP), Beijing, China,
17–20 September 2017.

-use multistage training scheme to overcome the overfitting problem

paper: https://www.mdpi.com/2076-3417/9/3/427

-max pooling prevents overfitting 
