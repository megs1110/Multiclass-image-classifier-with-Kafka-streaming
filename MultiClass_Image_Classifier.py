from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
import matplotlib
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils import np_utils
from imutils import build_montages
import numpy as np
import cv2
from keras.models import load_model 


class CNN:
    """Builds a CNN Model 
     
    Args:
      height (int): Height of input images
      width (int): Width of input images
      depth (int): Number of Channels
    
    Returns:
      Model : Keras model
    """  
    
    @staticmethod
    def build(height, width, depth, classes):
      # initialize the model along with the input shape
      # to be "channels last" and the channels dimension itself
      model = Sequential()
      inputShape = (height, width, depth)
      chanDim = -1

      # 1st CONV -> RELU -> CONV -> RELU -> POOL layer set
      model.add(Conv2D(32, (3,3), padding='same', input_shape=inputShape))
      model.add(Activation('relu'))
      model.add(BatchNormalization(axis=chanDim))
      model.add(Conv2D(32, (3,3), padding='same'))
      model.add(Activation('relu'))
      model.add(BatchNormalization(axis=chanDim))
      model.add(MaxPooling2D(pool_size=(2,2)))
      model.add(Dropout(0.25))

      # 2nd CONV -> RELU -> CONV -> RELU -> POOL layer set
      model.add(Conv2D(64, (3,3), padding='same', input_shape=inputShape))
      model.add(Activation('relu'))
      model.add(BatchNormalization(axis=chanDim))
      model.add(Conv2D(64, (3,3), padding='same'))
      model.add(Activation('relu'))
      model.add(BatchNormalization(axis=chanDim))
      model.add(MaxPooling2D(pool_size=(2,2)))
      model.add(Dropout(0.25))

      # 1st (and only) set of FC => RELU layers
      model.add(Flatten())
      model.add(Dense(512))
      model.add(Activation("relu"))
      model.add(BatchNormalization())
      model.add(Dropout(0.25))

      # softmax classifier
      model.add(Dense(classes))
      model.add(Activation("softmax"))

      # return the constructed network architecture

      return model


class TrainClassifier:
    """To train the CNN based classifier
    
    Args:
        X_train (numpy matrix): Training data inputs
        y_train (numpy matrix): Training data labels
        X_test (numpy matrix): Test Data inputs
        y_Test (numpy matrix): Test data labels
        n_classes (int): Number of Labels
        labelNames (List): Label or class names 
        
    Attributes:
        X_train (numpy matrix): Training data inputs
        y_train (numpy matrix): Training data labels
        X_test (numpy matrix): Test Data inputs
        y_Test (numpy matrix): Test data labels
        n_classes (int): Number of Labels
        labelNames (List): Label or class names 
    """
    def __init__(self, X_train, y_train, X_test, y_test, n_classes, labelNames):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_classes = n_classes
        self.labelNames = labelNames

    def Train(self):
        """Train the CNN based classifier 
        The hyperparameters are fixed for now
        
        Returns:
            model: Trained keras model
        """
        
        # Initialize EPOCHS, LR_RATE, BATCH_SIZE
        NUM_EPOCHS = 5
        INIT_LR = 1e-2
        BATCH_SIZE = 32
          
        ## Preprocess dataset + prepare data

        # Scale data to the frame of [0, 1]
        X_train = self.X_train.astype("float32") / 255.0
        X_test = self.X_test.astype("float32") / 255.0

        #one=hot encode the training and test labels
        y_train = np_utils.to_categorical(self.y_train, 10)
        y_test = np_utils.to_categorical(self.y_test, 10)

        #Get height,width and depth of input images
        height, width, depth = X_train[0].shape
        
        # Initialize the optimizer and model
        print("Compiling Model...")
        opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)
        model = CNN.build(width, height, depth, self.n_classes)
        model.compile(loss="categorical_crossentropy", optimizer=opt, 
                                           metrics=['accuracy'])

        # Train The model
        print("Training model...")
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                           batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)

        model.save('generator/trained_model/model.h5') 
        np.save('generator/test_set/test.npy',X_test)
        np.save('generator/test_set/labels.npy',np.array(self.labelNames))
        
        # Final evaluation of the model
        scores = model.evaluate(X_train, y_train, verbose=1)
        print("Train Score: %.2f%%" % (scores[0]*100))
        print("Train Accuracy: %.2f%%" % (scores[1]*100))

        ############## Final evaluation of the model For Test Set
        score = model.evaluate(X_test, y_test, verbose=1)
        print("Test Score: %.2f%%" % (score[0]*100))
        print("Test Accuracy: %.2f%%" % (score[1]*100))

        return model    
