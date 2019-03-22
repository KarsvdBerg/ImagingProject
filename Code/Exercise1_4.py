import numpy as np 
import matplotlib.pyplot as plt 

allimages = []
#img0 = []
#img1 = []
for i in range(1,21):
    img_name0 = 'images/0/' + str(i) + '.jpg'
    allimages.append(img_name0) 
    img_name1 = 'images/1/' + str(i) + '.jpg'
    allimages.append(img_name1)
    allimages.sort()

fig = plt.figure(figsize=(5,50))
for i in range(1,41):
    image = plt.imread(allimages[i-1])
    plt.subplot(20,2,i)
    plt.imshow(image)
    plt.axis('off')

fig.suptitle('With and without metastases', fontsize=14, fontweight='bold')
plt.show()
            

