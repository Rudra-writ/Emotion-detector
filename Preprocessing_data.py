from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
classes=os.listdir('D:/emotions/images/train')
labels=[i for i in range(len(classes))]

label_dict=dict(zip(classes,labels)) 
img_size=55
images=[]
labels=[]


for class_iterator in classes:
    class_path = os.path.join('Your path to train data set',class_iterator)
    contents = os.listdir(class_path)
        
    for content in contents:
        content_path = os.path.join(class_path,content)
        img=cv2.imread(content_path)
        gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized=cv2.resize(gray_scale,(img_size,img_size))
        images.append(resized)
        labels.append(label_dict[class_iterator])
           
images=np.array(images)/255.0
images=np.reshape(images,(images.shape[0],img_size,img_size,1))
print(images.shape)
plt.imshow(images[25000
                 ], cmap='gray')
labels=np_utils.to_categorical(np.array(labels))
np.save('images',images)
np.save('labels',labels)
