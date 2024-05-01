#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 14 09:15:03 2022

@author: federicovitro
"""
# Basic libraries
import numpy as np

# Neural Network
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

# Model evaluation
from sklearn.metrics import mean_squared_error

def cost_function_NNregress_complex_ES(th_log, X_train, X_val, y_train, y_val):
    
    # Faccio una copia di train e validation
    X_train0 = X_train.copy()
    X_val0 = X_val.copy()
    
    # Trasformo i parametri di riscalamento da scala log a scala reale
    th = 10**th_log
    
    # Riscalo train e val
    X_train0 = X_train0*th
    X_val0 = X_val0*th
    
    # Define NN
    nn_regressor = Sequential()
    
    # The Input Layer
    nn_regressor.add(Dense(128, kernel_initializer='normal', input_dim = X_train0.shape[1], activation='relu'))
    
    # The Hidden Layers
    nn_regressor.add(Dense(256, kernel_initializer='normal',activation='relu'))
    nn_regressor.add(Dense(256, kernel_initializer='normal',activation='relu'))
    nn_regressor.add(Dense(256, kernel_initializer='normal',activation='relu'))
    
    # The Output Layer :
    nn_regressor.add(Dense(1, kernel_initializer='normal',activation='linear'))
    
    # Compile the network
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    nn_regressor.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mean_absolute_error'])
    # Early stopping
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, mode='min')

    # Train the model
    nn_regressor.fit(X_train0, y_train, validation_data=(X_val0, y_val), epochs=10000, callbacks=[callback], verbose=0)
    
    # Computing the Root Mean Square Error
    rmse_nn_val = (np.sqrt(mean_squared_error(y_val, nn_regressor.predict(X_val0))))
 
    return rmse_nn_val

def cost_function_NNregress_simple_ES(th_log, X_train, X_val, y_train, y_val):
    
    # Faccio una copia di train e validation
    X_train0 = X_train.copy()
    X_val0 = X_val.copy()
    
    # Trasformo i parametri di riscalamento da scala log a scala reale
    th = 10**th_log
    
    # Riscalo train e val
    X_train0 = X_train0*th
    X_val0 = X_val0*th
    
    # Define NN
    nn_regressor = Sequential()
    
    # The Input Layer
    nn_regressor.add(Dense(X_train0.shape[1], kernel_initializer='normal', input_dim = X_train0.shape[1], activation='relu'))
    
    # The Hidden Layers
    nn_regressor.add(Dense(100, kernel_initializer='normal',activation='relu'))
    
    # The Output Layer :
    nn_regressor.add(Dense(1, kernel_initializer='normal',activation='linear'))
    
    # Compile the network
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    nn_regressor.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mean_absolute_error'])
    # Early stopping
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=200, mode='min')

    # Train the model
    nn_regressor.fit(X_train0, y_train, validation_data=(X_val0, y_val), epochs=10000, callbacks=[callback], verbose=0)
    
    # Computing the Root Mean Square Error
    rmse_nn_val = (np.sqrt(mean_squared_error(y_val, nn_regressor.predict(X_val0))))
 
    return rmse_nn_val

