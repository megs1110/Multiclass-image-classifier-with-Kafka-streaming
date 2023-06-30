import os
from time import sleep
import json
import keras
import numpy as np
import tensorflow as tf
from keras.models import load_model
from kafka import KafkaProducer

TRANSACTIONS_TOPIC = os.environ.get('TRANSACTIONS_TOPIC')
KAFKA_BROKER_URL = os.environ.get('KAFKA_BROKER_URL')
TRANSACTIONS_PER_SECOND = float(os.environ.get('TRANSACTIONS_PER_SECOND'))
SLEEP_TIME = 1 / TRANSACTIONS_PER_SECOND


if __name__ == '__main__':
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER_URL,
        # Encode all values as JSON
        value_serializer=lambda value: json.dumps(value).encode(),
    )
    X_test = np.load('test_set/test.npy')
    X_test = X_test.astype("float32") / 255.0
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
    labels = np.load('test_set/labels.npy')
    def createtransaction(X_test):
        model = load_model('trained_model/model.h5')
        for i in range(X_test.shape[0]):
            test = X_test[i].reshape(1, X_test[i].shape[0],X_test[i].shape[1],X_test[i].shape[2])
            yield {'test_sample':i, 'label': str(labels[int(np.argmax(model.predict(test)))])}

    a =  createtransaction(X_test)    

    while True:
        transaction: dict = next(a)
        producer.send(TRANSACTIONS_TOPIC, value=transaction)
        print(transaction)  # DEBUG
        sleep(SLEEP_TIME)
