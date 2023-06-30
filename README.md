# Multiclass-image-classifier-with-Kafka-streaming
[![Kafka](https://img.shields.io/badge/streaming_platform-kafka-black.svg?style=flat-square)](https://kafka.apache.org)
[![Docker Images](https://img.shields.io/badge/docker_images-confluent-orange.svg?style=flat-square)](https://github.com/confluentinc/cp-docker-images)
[![Python](https://img.shields.io/badge/python-3.5+-blue.svg?style=flat-square)](https://www.python.org)

This project is a combination of two parts: a multiclass image classifier and a Kafka stream classifier.

## Multiclass Image Classifier

The multiclass image classifier is trained using a dataset of fashion images. It uses the Keras library and the Fashion MNIST dataset, which consists of 60,000 training images and 10,000 test images. The classifier is trained to predict the type of clothing item based on the images, with 10 different labels such as "top," "trouser," "pullover," etc. The training process reshapes the image data and then trains the classifier using the provided training and test data.

To run the multiclass image classifier, follow these steps:

- Import the required packages:
```python
import MultiClass_Image_Classifier as MIC
from keras.datasets import fashion_mnist
import numpy as np
```

- Load the dataset and labels
```python
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
labels = ["top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]
```
- Reshape the training matrices in the following format: (number of samples, height, width, channels)
```python
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)) 
```
- Begin training
```python
MIC.TrainClassifier(X_train,y_train, X_test, y_test,10,labels).Train()
```
Note: The model is saved at 'generator/trained_model' in h5 format with the name 'model.h5'. The test set and the labels are saved as numpy matrices at 'generator/test_set' with names 'test_set.npy' and 'labels.npy' respectively.

## Kafka Stream Classifier
The Kafka stream classifier is designed to work with a Kafka cluster, which is set up using Docker and Docker Compose. The cluster allows communication between the Kafka broker and the applications. The stream classifier consists of a transaction generator and a detector. The kafka stream classifier is fully containerised. You would need [Docker](https://docs.docker.com/install/) and [Docker Compose](https://docs.docker.com/compose/) to run it.

To run the Kafka stream classifier, you need to set up the Docker network, start the Kafka cluster, build and start the transaction generator and detector containers. Then, you can use the provided command to view the stream of transactions and their predicted labels. Finally, you can stop the containers and remove the Docker network when you're done., follow these steps:

### 1. Create a Docker network called network to enable communication between the Kafka cluster and the apps:

     $ docker network create network

### 2. Spin up the local single-node Kafka cluster:
     
     $ docker-compose -f docker-compose.kafka.yml up -d
    
### 3. Start the transaction generator and the detector (which uses the saved classifier model to predict labels):

    $ docker-compose build
    $ docker-compose up -d

## Usage

Show a stream of transactions as follows:

```bash
$ docker-compose -f docker-compose.kafka.yml exec broker kafka-console-consumer --bootstrap-server localhost:9092 --topic queueing.transaction
```

Example transaction messages for the stream:

```bash
{"test_sample": 9977, "label": "top"}
{"test_sample": 9978, "label": "ankle boot"}
{"test_sample": 9979, "label": "top"}
{"test_sample": 9980, "label": "top"}
{"test_sample": 9981, "label": "top"}
{"test_sample": 9982, "label": "bag"}
{"test_sample": 9983, "label": "trouser"}
{"test_sample": 9984, "label": "dress"}
{"test_sample": 9985, "label": "pullover"}
{"test_sample": 9986, "label": "sneaker"}
{"test_sample": 9987, "label": "sandal"}
{"test_sample": 9988, "label": "bag"}
{"test_sample": 9989, "label": "shirt"}
{"test_sample": 9990, "label": "sandal"}\
```
## Closing

To stop the transaction generator and detector:

```bash
$ docker-compose down
```

To stop the Kafka cluster (can use `down` to also remove contents of the topics):

```bash
$ docker-compose -f docker-compose.kafka.yml stop
```

To remove the Docker network:

```bash
$ docker network rm kafka-network
```
Please make sure you have Docker and Docker Compose installed before running the Kafka stream classifier.

In summary, this project trains a multiclass image classifier using a fashion dataset and saves the model. It then utilizes a Kafka stream classifier to process a stream of transactions and predict labels for the test samples using the saved model.
