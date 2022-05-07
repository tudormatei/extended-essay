import pickle
import pandas as pd
import numpy as np
import matplotlib as plt
import random
from sklearn.utils import shuffle
from matplotlib.image import imread

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

from subprocess import check_output
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_model():
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=image_shape))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(43, activation='sigmoid'))  

    model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001, beta_1=0.9, beta_2=0.999, amsgrad=False), metrics=['accuracy'])

    return model


img_height = 30
img_width = 30
channels = 3
image_shape = (img_height, img_width, channels)


def load_data():
    train_path = './/data//train'
    test_path = './/data//test'

    image_gen = ImageDataGenerator()
    train_image_gen = image_gen.flow_from_directory(train_path, target_size=image_shape[:2], color_mode='rgb', batch_size=16)
    test_image_gen = image_gen.flow_from_directory(test_path, target_size=image_shape[:2], color_mode='rgb', batch_size=16, shuffle=False)
    
    return (train_image_gen, test_image_gen)


if __name__ == '__main__':
    train_image_gen, test_image_gen = load_data()

    model = create_model()
    print(model.summary())

    history = model.fit(train_image_gen, epochs=20)
    