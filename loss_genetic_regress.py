#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 14 09:15:03 2022

@author: federicovitro
"""
# Basic libraries
import numpy as np

# regression models
from sklearn.ensemble import RandomForestRegressor

# Neural Network
from keras.models import Sequential
from keras.layers import Dense

# Model evaluation
from sklearn.metrics import mean_squared_error

def cost_function_NNregress_complex_noES(th_log, X_train, X_val, y_train, y_val):
    
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
    
    # Compile the network :
    # lr = 0.01 #leaning rate for Stochastic Gradient Descent
    # opt = keras.optimizers.SGD(learning_rate=lr, momentum=0.95)
    nn_regressor.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    
    # # Define early stopping
    # es = EarlyStopping(monitor='val_loss', mode='min', patience=200)
    
    # Train the model
    nn_regressor.fit(X_train0, y_train, epochs=500, batch_size=32, verbose=0) # callbacks=[es]
    
    # Computing the Root Mean Square Error
    rmse_nn_val = (np.sqrt(mean_squared_error(y_val, nn_regressor.predict(X_val0))))
 
    return rmse_nn_val

# def cost_function_LinReg(th, X_train, X_val, Y_train, Y_val):
    
#     N_features = np.shape(X_train)[1]
    
#     X_train0 = X_train.copy()
#     X_val0 = X_val.copy()
#     Y_train0 = Y_train.copy()
    
#     for i in range(N_features):
#         X_train0[:,i] = th[i]*X_train0[:,i]
#         X_val0[:,i] = th[i]*X_val0[:,i]

#     # create model
#     lin_model = linear_model.LinearRegression()
#     # fit model
#     lin_model.fit(X_train0, Y_train0)
#     # make prediction
#     pred = lin_model.predict(X_val0)
#     rnd_pred = pred[:,1]
#     # compute error
#     MSE = mean_squared_error(Y_val[:,1], rnd_pred)
 
#     return MSE

# def cost_function_tree(th, X_train, X_val, Y_train, Y_val):
    
#     N_features = np.shape(X_train)[1]
    
#     X_train0 = X_train.copy()
#     X_val0 = X_val.copy()
#     Y_train0 = Y_train.copy()
    
#     for i in range(N_features):
#         X_train0[:,i] = th[i]*X_train0[:,i]
#         X_val0[:,i] = th[i]*X_val0[:,i]

#     # create model
#     tree_model = DecisionTreeRegressor(max_depth=5)
#     # fit model
#     tree_model.fit(X_train0, Y_train0)
#     # make prediction
#     pred = tree_model.predict(X_val0)
#     rnd_pred = pred[:,1]
#     # compute error
#     MSE = mean_squared_error(Y_val[:,1], rnd_pred)
 
#     return MSE

def cost_function_random_forest(th_log, X_train, X_val, y_train, y_val):
    
    # Faccio una copia di train e validation
    X_train0 = X_train.copy()
    X_val0 = X_val.copy()
    
    # Trasformo i parametri di riscalamento da scala log a scala reale
    th = 10**th_log
    
    # Riscalo train e val
    X_train0 = X_train0*th
    X_val0 = X_val0*th
    
    random_seed = 2398745
    # Evaluate the model
    regressor_rf = RandomForestRegressor(n_estimators = 500, random_state = random_seed)
    
    regressor_rf.fit(X_train0,y_train)
    
    # Computing the Root Mean Square Error
    rmse_rf_test = (np.sqrt(mean_squared_error(y_val, regressor_rf.predict(X_val0))))
    
    return rmse_rf_test