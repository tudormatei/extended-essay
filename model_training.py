import pickle
import pandas as pd
import numpy as np
import matplotlib as plt
import random
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

from subprocess import check_output
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import EarlyStopping

def create_model():
    image_shape = x_train[1:]

    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=image_shape))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(43, activation='sigmoid'))  

    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(0.001, beta_1=0.9, beta_2=0.999, amsgrad=False), metrics=['accuracy'])

    return model

x_train = None
y_train = None
x_validation = None
y_validation = None
x_test = None
y_test = None

def load_data():
    with open("./traffic-signs-data/train.p", mode='rb') as training_data:
        train = pickle.load(training_data)
    
    with open("./traffic-signs-data/valid.p", mode='rb') as validation_data:
        validation = pickle.load(validation_data)

    with open("./traffic-signs-data/test.p", mode='rb') as testing_data:
        test = pickle.load(testing_data)

    x_train, y_train = train['features'], train['labels']
    x_validation, y_validation = validation['features'], validation['labels']
    x_test, y_test = test['features'], test['labels']

    x_train, y_train = shuffle(x_train, y_train)

    normal_and_gray()

def normal_and_gray():
    x_train_gray = np.sum(x_train/3, axis=3, keepdims=True)
    x_test_gray = np.sum(x_test/3, axis=3, keepdims=True)
    x_validation_gray = np.sum(x_validation/3, axis=3, keepdims=True)

    x_train = ((x_train_gray - 128)/128)
    x_test = ((x_test_gray - 128)/128)
    x_validation = ((x_validation_gray - 128)/128)

if __name__ == '__main__':
    load_data()

    model = create_model()

    print(model.summary())
    batch_size = 500
    early_stop = EarlyStopping(monitor='val_loss', patience=2)

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=50, verbose=1, validation_data=(x_validation, y_validation))
    