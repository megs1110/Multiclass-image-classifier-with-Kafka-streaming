import MultiClass_Image_Classifier as MIC
from keras.datasets import fashion_mnist
import numpy as np


#Sample training of fashion MNIST data for classification task
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
labels = ["top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)) 
MIC.TrainClassifier(X_train,y_train, X_test, y_test,10,labels).Train()
