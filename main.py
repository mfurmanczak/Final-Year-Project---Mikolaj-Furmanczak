import os
from datetime import datetime

import pandas as pd
import numpy as np
import librosa

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

from keras.utils import np_utils, to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot as plt
import seaborn as sns


def create_model():
    model = Sequential()
    model.add(Conv2D(
        filters=24,
        kernel_size=(kernel_filter_pixel+2, kernel_filter_pixel+2),
        input_shape=(128, 128, 1),
        kernel_regularizer=regularizers.l2(kernel_regularizer)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=3))
    model.add(Activation('relu'))

    model.add(Conv2D(
        filters=36,
        kernel_size=(kernel_filter_pixel+1, kernel_filter_pixel+1),
        padding='valid',
        kernel_regularizer=regularizers.l2(kernel_regularizer)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Activation('relu'))

    model.add(Conv2D(
        filters=48,
        kernel_size=(kernel_filter_pixel, kernel_filter_pixel),
        padding='valid'))
    model.add(Activation('relu'))

    model.add(GlobalAveragePooling2D())

    model.add(Dense(60))
    model.add(Activation('relu'))
    model.add(Dropout(droprate))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001), loss='categorical_crossentropy', metrics=["accuracy"])


    return model


def init_data_aug():
    train_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        fill_mode='constant',
        cval=-80.0,
        width_shift_range=0.1,
        height_shift_range=0.0)

    val_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        fill_mode='constant',
        cval=-80.0)

    return train_datagen, val_datagen


def train_test_split(k_fold, data, X_dimensions=(128, 128, 1)):
    X_train = np.stack(data[data.fold != k_fold].melspectrogram.to_numpy())
    X_test = np.stack(data[data.fold == k_fold].melspectrogram.to_numpy())

    y_train = data[data.fold != k_fold].label.to_numpy()
    y_test = data[data.fold == k_fold].label.to_numpy()

    XX_train = X_train.reshape(X_train.shape[0], *X_dimensions)
    XX_test = X_test.reshape(X_test.shape[0], *X_dimensions)

    yy_train = to_categorical(y_train)
    yy_test = to_categorical(y_test)

    return XX_train, XX_test, yy_train, yy_test


def train(k_fold, data, epochs, batch_size):
    X_train, X_test, y_train, y_test = train_test_split(k_fold, data)

    train_datagen, val_datagen = init_data_aug()

    train_datagen.fit(X_train)
    val_datagen.fit(X_train)

    model = create_model()

    early_stop = EarlyStopping(
        monitor='loss', patience=early_stop_patience, verbose=1)

    score = model.evaluate(val_datagen.flow(
        X_test, y_test, batch_size=batch_size), verbose=0)
    print("Pre-training accuracy: %.4f%%\n" % (100 * score[1]))

    start = datetime.now()
    history = model.fit(train_datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(X_train) / batch_size,
                        epochs=epochs,
                        validation_data=val_datagen.flow(
                            X_test, y_test, batch_size=batch_size),
                        callbacks=[early_stop])
    end = datetime.now()
    print("Training completed in time: ", end - start, '\n')

    return history


###### HYPER PARAMETERS ######
model_num = 4

batch_size = 32
epochs = 100
kernel_filter_pixel = 3
droprate = 0.5
kernel_regularizer = 0.01
early_stop_patience = 15


k_fold = 1
repeat = 1
###### HYPER PARAMETERS ######


def for_every_fold(data, kfold):
    histories = []
    for j in range(repeat):
        print('-'*80)
        print("\n({})\n".format(j+1))

        history = train(kfold, data, epochs, batch_size)
        histories.append(history)

    for i, history in enumerate(histories):
        history_df = pd.DataFrame(history.history)
        history_df.to_csv('history/Model ' + str(model_num) + '/history_{}.csv'.format(kfold), index=False)


def main():
    us8k_df = pd.read_pickle("us8k_df.pkl")

    for i in range(1, 11):
        for_every_fold(us8k_df, i)
 
if __name__ == '__main__':
    main()
