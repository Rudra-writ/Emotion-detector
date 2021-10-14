import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import models,layers
import matplotlib.pyplot as plt

images=np.load('images.npy')
labels=np.load('labels.npy')

X_train,X_test,Y_train,Y_test=train_test_split(images,labels,test_size=0.1)

cnn = models.Sequential([
    layers.Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', padding = 'same', input_shape = (55,55,1)),
    layers.BatchNormalization(),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0.25),
    layers.Conv2D(filters = 256, kernel_size = (3,3), padding = 'same', activation = 'relu'),
    layers.BatchNormalization(),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0.25),
    layers.Conv2D(filters = 512, kernel_size = (3,3), padding = 'same', activation = 'relu'),
    layers.BatchNormalization(),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0.25),
   
   
    
    
    layers.Flatten(),
   
    layers.Dense(256, activation = 'relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.25),
    layers.Dense(512, activation = 'relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.25),
    
    layers.Dense(7, activation='softmax')
    
    ])


cnn.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.0002),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

history = cnn.fit(X_train,Y_train,epochs=12,validation_split=0.2)
cnn.evaluate(X_test,Y_test)
cnn.save('emotion detector')
print(history.history.keys())
plt.plot(history.history['accuracy'], c='r')
plt.plot(history.history['val_accuracy'], c='b')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'],c='r')
plt.plot(history.history['val_loss'], c='b')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()