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

from keras.utils import to_categorical

from classifier import Att_LSTMClassifier
from utils import convert_y

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
    met = np.array(df.iloc[:,2]).reshape(len(np.array(df.iloc[:,2])),1)
    time_series = np.array(df.iloc[:,7:])
    time_series = minmax_scale(time_series, axis=1)
    X_activity = pd.DataFrame(np.hstack((met, time_series)))
    y = df['event']

    X_train, X_test, y_train, y_test = train_test_split(X_activity, y, test_size=0.3, random_state=42)
    met_train = X_train.iloc[:,0]
    met_test = X_test.iloc[:,0]
    met_train = to_categorical(met_train)
    met_test = to_categorical(met_test)

    X_activity_train = X_train.iloc[:,1:]
    X_activity_test = X_test.iloc[:,1:]
    X_activity_train = X_activity_train.to_numpy().reshape(len(X_activity_train),-1,1)
    X_activity_test = X_activity_test.to_numpy().reshape(len(X_activity_test),-1,1)

    y_train = convert_y(y_train)
    y_test = convert_y(y_test)

    # modeling
    lstm = Att_LSTMClassifier(nb_epochs = epoch, verbose = True)
    lstm.fit(X_activity_train, met_train, y_train, X_activity_test, met_test, y_test)


if __name__ == "__main__":
    main()
