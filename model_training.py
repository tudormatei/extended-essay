import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from tensorflow.keras.callbacks import EarlyStopping

def create_model(image_shape):
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=image_shape))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=image_shape))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=image_shape))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossenthropy', optimizer='adam',metrics=['accuracy'])

    return model

def main():
    print('Main up and running!')
    model = create_model((128, 128, 3))
    print(model.summary())

    early_stop = EarlyStopping(monitor='val_loss', patience=2)
    batch_size = 16
    

if __name__ == '__main__':
    main()