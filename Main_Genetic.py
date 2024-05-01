#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 17:03:33 2022

@author: federicovitro
"""

# Basic libraries
import numpy as np
import pandas as pd

# Others
import time
#from loss_genetic_regress import cost_function_NNregress_complex_noES
from loss_genetic_EStopping import cost_function_NNregress_complex_ES, cost_function_NNregress_simple_ES
import ga
from ypstruct import structure

# Preprocessing
from sklearn.preprocessing import StandardScaler

# Grab Currrent Time Before Running the Code
start = time.time()

# # init_weights = 'init_weights' # scommenta per inizializzare i pesi+bias della N
# # Import datas
# D = pd.read_csv("/Users/federicovitro/Desktop/Tesi_Magistrale/Step_8/Dataset_skew.csv")
# D = D.loc[:,'x0':]

# # train/val/test split (hold out procedure)
# random_seed = 2398745
# train, test = train_test_split(D, test_size = 0.3, random_state = random_seed)
# test, val = train_test_split(test, test_size = 0.5, random_state = random_seed)

# train.to_csv("/Users/federicovitro/Desktop/Tesi_Magistrale/Step_9/train_NN_set.csv")
# val.to_csv("/Users/federicovitro/Desktop/Tesi_Magistrale/Step_9/val_NN_set.csv")
# test.to_csv("/Users/federicovitro/Desktop/Tesi_Magistrale/Step_9/test_NN_set.csv")

train = pd.read_csv("/Users/federicovitro/Desktop/Tesi_Magistrale/Step_9/train_base.csv")
val = pd.read_csv("/Users/federicovitro/Desktop/Tesi_Magistrale/Step_9/val_base.csv")
test = pd.read_csv("/Users/federicovitro/Desktop/Tesi_Magistrale/Step_9/test_base.csv")

# I/O definition
X_train = train.loc[:,'x0':'x12']
y_train = train['t']
X_val = val.loc[:,'x0':'x12']
y_val = val['t']
X_test = test.loc[:,'x0':'x12']
y_test = test['t']

#%% Center the data
# Xc_train = X_train.subtract(X_train.mean())
# Xc_val = X_val.subtract(X_val.mean())
# Xc_test = X_test.subtract(X_test.mean())

#%% Standardize the data
scaler = StandardScaler()
Xs_train = scaler.fit_transform(X_train)
Xs_val = scaler.transform(X_val)
Xs_test = scaler.transform(X_test)

columns = ['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12']
Xs_train = pd.DataFrame(Xs_train, columns=columns)
Xs_val = pd.DataFrame(Xs_val, columns=columns)
Xs_test = pd.DataFrame(Xs_test, columns=columns)

#%% Problem definition
N_decision_var = X_train.shape[1]
problem = structure()
#problem.costfunction = cost_function_NNregress_complex_noES # loss: RMSE
problem.costfunction = cost_function_NNregress_complex_ES # loss: RMSE
problem.nvar = N_decision_var # numero di decision variables
problem.varmin = -3 # valore minimo che possono assumere le decision variables (in scala log)
problem.varmax = 3 # valore massimo che possono assumere le decision variables (in scala log)

problem.xtrain = Xs_train
problem.xval = Xs_val
problem.ytrain = y_train
problem.yval = y_val
# problem.init_weights = init_weights

#%% Genetic Algorithm (GA) parameters
params = structure()
params.maxit = 100 # numero massimo di iterazioni
params.npop = 20 # numero di elementi nella popolazione valutata
params.pc = 1 # pc è la proporzione dei figli rispetto alla popolazione iniziale (proporzione children)
params.gamma = 0.1
params.mu = 0.2 # media e varianza della distribuzione normale 
params.sigma = 0.1

#%% Run GA
out = ga.run(problem, params)

# Grab Currrent Time After Running the Code
end = time.time()

#Subtract Start Time from The End Time
total_time_s = end - start
print("\ntotal time = "+ str(total_time_s)+" [s]")
total_time_h = (end - start)/3600
print("\ntotal time = "+ str(total_time_h)+" [h]")
total_time = np.zeros((1,2))
total_time[0,0] = total_time_s
total_time[0,1] = total_time_h
np.savetxt('/Users/federicovitro/Desktop/Tesi_Magistrale/Step_10/total_time_complexNN_siEStopping_Standardized.csv', total_time, delimiter=',')

#%% Save Results + dataset

columns = ['th0', 'th1', 'th2', 'th3', 'th4',  'th5', 'th6', 'th7', 'th8', 'th9', 'th10', 'th11', 'th12', 'best_cost']
pop_mat = np.zeros((20, 14))

for i in range(0,20):
    for j in range(0,14):
        if j!=13:
            pop_mat[i,j] = out.pop[i].position[j]
        else:
            pop_mat[i,j] = out.pop[i].cost

pop_df = pd.DataFrame(pop_mat, columns = columns)
pop_df.to_csv('/Users/federicovitro/Desktop/Tesi_Magistrale/Step_10/population_complexNN_siEStopping_Standardized.csv')

best_sol = np.zeros((1,14))

for i in range(0,14):
    if i!=13:
        best_sol[0,i] = out.bestsol.position[i]
    else:
        best_sol[0,i] = out.bestsol.cost

bestsol_df = pd.DataFrame(best_sol, columns = columns)
bestsol_df.to_csv('/Users/federicovitro/Desktop/Tesi_Magistrale/Step_10/bestsol_complexNN_siEStopping_Standardized.csv')

