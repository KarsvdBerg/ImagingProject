# Assignment 1
## Exercise 1
The presence of metastases is an important component of breast cancer staging [1]. Pathologist evaluate this by examination of histological preparations of lymph nodes. Currently, the lymph nodes are assessed manually. This is a rather complicated method that can result in a missed metastases. Consequently, this will result in an inadequate treatment for the patient. Introducing whole-slide images, will optimize the assessment of metastases [2]. It will also make the detection of metastases more accurate and objective. 
## Exercise 2
The current role of the pathologist is reviewing the biological tissue. It is a labor intensive and error-prone process [3]. By introducing whole-slide imaging, the workflow in a pathology lab will change. The detection of the metastases by deep learning will increase the speed and accuracy of the detection. It will save the pathologists a lot of time, they will now have a more evaluating role, instead of a determining role in the detection of metastases.
## Exercise 3
A neural network that works with small images for registration is Fully Convolutional Network. This is an neural network that consists of only convolutional layers. It consists of layers that produce outputs for arbitrary input sizes, this enables the model to be trained on small images and then applied to larger images such as whole slide images. [4] In the CAMELYON16 competition this model was used in 2 different methods from two different research groups, HTW-Berlin and CULab III. CULab III obtained the fourth highest FROC score (0.703) of all research teams participating in the competition.  Fully Convolutional NetworkShelhamer et al. explains how fully conventional networks (FNW) operate and where they originate from. Each layer output in a FNW is a convolutional network with a 3D array of size h x w x d, where h and w are spatial dimensions, and d is the feature dimension. The first layer is the image, with pixel size h x w and d features. Location in the higher layers correspond to the locations in the image they are connected to, which are called receptive field. A real-valued loss function composed with a FCN defines a task. The stochastic gradient decent computed on whole images will be the same as the stochastic gradient decent on the receptive fields, which are used as minibatches. When these receptive fields overlap significantly, both feedforward computation and backpropagation are much more efficient when computed layer-by-layer over an entire image.[5]


**References**

[1] Donegan, W.L., 1997. Tumor-related prognostic factors for breast cancer. CA: a cancer journal for clinicians, 47.

[2] Madabhushi, A. and Lee, G., 2016. Image analysis and machine learning in digital pathology: Challenges and opportunities. Medical Image Analysis, 33.

[3] Liu, Y., Gadepalli, K., Norouzi, M., Dahl, G.E., Kohlberger, T. et al. 2017. Detecting Cancer Metastases on Gigapixel Pathalogy Images. 

[4] Ehteshami Bejnordi, B., Veta, M., Johannes van Diest, P., van Ginneken, B., Karssemeijer, N., Litjens, G., ... Venâncio, R. (2017). Diagnostic Assessment of Deep Learning Algorithms for Detection of Lymph Node Metastases in Women With Breast Cancer. JAMA, 318(22), 2199–2210. https://doi.org/10.1001/jama.2017.14585

[5] Shelhamer, E., Long, J., & Darrell, T. (2017). Fully Convolutional Networks for Semantic Segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 39(4), 640–651. https://doi.org/10.1109/TPAMI.2016.2572683

## Exercise 4
There are histopathologic scan with an purple staining. Within one classification, the images differ a lot. By just looking at them no clear signs of metastases could be found. No clear distoinction could be made when looking at the two different classes. The features pathologist look at are spatial arrangement of cells, morphometric characteristics of the nuclei, the presence of tubules, and the mitotic count [6]. 

**References**

[6] Karssemeijer, N. 2019. Breast Histology. http://www.diagnijmegen.nl/index.php/Breast_Histology
CODE:





```
## Exercise 5
An account has been made and our team name kaggle is : Imaging TU/e group 2
