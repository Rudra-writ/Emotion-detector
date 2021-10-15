# Real time Emotion detector
A software to detect emotions based on facial expression using CNN and OpenCV

The python files to be executed in the following sequences --> Preprocessing_data.py ---> Convolutional neural Network.py ----> Execution.py

The train dataset used, "Face expression recognition dataset" is a contribution from Mr. Jonathan Oheix and is downloadble from Kaggle. Link: https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset

The CNN model employed, takes over an hour to train. 

Early stopping has been used to reduce over fitting by limiting the epochs to 12. However the results achieved, for an abstract application of this type is still satisfying. Accuracy for some expressions like "fear" and "happy" being over 90%

Tweaking the hyper parameters (specially the number of epochs and learning rate) might fetch higher accuracies and performance. Would love to hear about it if someone volunteers!! :D
