#!/usr/bin/env python
# coding: utf-8

__author__ = 'Dan LIN'

import sys
sys.path.append('..')

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

import tensorflow as tf
import keras
from keras.utils import to_categorical
from tensorflow.keras.layers import Dropout, Dense, Input
from tensorflow.keras.models import Model

from classifier import T2V, ClassToken, transformer_encoder
from utils import recall, specificity, f1

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        required=True,
                        type=str,
                        help='Path to the data file. Please provide the absolute path.')
    parser.add_argument('--epoch',
                        required=True,
                        type=int,
                        help='The number of training epoch.')
    args = parser.parse_args()

    path = args.path
    epoch = args.epoch

    # prepare data
    print(path)
    df = pd.read_csv(path)
    df = df.dropna().reset_index()
    time_series = np.array(df.iloc[:,7:])
    X_activity = minmax_scale(time_series, axis=1)
    y = df['event']

    X_train_val, X_activity_test, y_train_val, y_test = train_test_split(X_activity, y, test_size=0.2, random_state=50)
    X_activity_train, X_activity_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=50)

    batch_size = 64
    print('Build model...')
    inputs = Input(shape=(14, 1))

    # Trans
    x = T2V(31)(inputs)
    x = keras.layers.Conv1D(filters=32, kernel_size=8, padding="same")(x)
    x = ClassToken(input_shape=(1, 32))(x)

    num_transformer_blocks = 2
    for _ in range(num_transformer_blocks):
        x, w = transformer_encoder(x, head_size=32, num_heads=2, ff_dim=128, dropout=0.1)

    cls_token = tf.keras.layers.Lambda(lambda v: v[:, 0], name="ExtractToken")(x)
    x = Dense(32, activation='relu')(cls_token)  # 128
    x = Dropout(0.1)(x)
    predictions = keras.layers.Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=predictions)

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-06),  # 'adam',
                  metrics=['accuracy', recall, specificity, f1])
    model.summary()

    print('Train...')
    history = model.fit(X_activity_train, y_train,
                        batch_size=batch_size,
                        epochs=epoch,
                        validation_data=(X_activity_val, y_val))

    score, train_acc, train_sen, train_spec, train_f1 = model.evaluate(X_activity_train, y_train,
                                                                       batch_size=batch_size)
    score, val_acc, val_sen, val_spec, val_f1 = model.evaluate(X_activity_val, y_val,
                                                               batch_size=batch_size)
    score, test_acc, test_sen, test_spec, test_f1 = model.evaluate(X_activity_test, y_test,
                                                                   batch_size=batch_size)

    print('Train acc:', train_acc)
    print('Train sen:', train_sen)
    print('Train spec:', train_spec)
    print('Train f1:', train_f1)
    print('Val acc:', val_acc)
    print('Val sen:', val_sen)
    print('Val spec:', val_spec)
    print('Val f1:', val_f1)
    print('Test acc:', test_acc)
    print('Test sen:', test_sen)
    print('Test spec:', test_spec)
    print('Test f1:', test_f1)

if __name__ == "__main__":
    main()
