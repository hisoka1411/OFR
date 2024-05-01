#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 10:43:11 2022

@author: federicovitro
"""

import numpy as np
from ypstruct import structure

def run(problem, params):
    
#%% STEP:
    # 1- Initialization

    # Problem information
    costfunction = problem.costfunction
    nvar = problem.nvar
    varmin = problem.varmin
    varmax = problem.varmax
    
    X_train = problem.xtrain
    X_val = problem.xval
    y_train = problem.ytrain
    y_val = problem.yval
    # init_weights = problem.init_weights
    
    # Parameters
    maxit = params.maxit
    npop = params.npop
    pc = params.pc # pc è la proporzione dei figli rispetto alla popolazione iniziale
    nc = int(np.round(pc*npop/2)*2) # nc è il numero dei figli (children)
    # l'output della funzione round è un integer, quindi, 2*int = numero pari. nc//2 è un numero pari a sua volta
    gamma = params.gamma
    mu = params.mu
    sigma = params.sigma
    
    # Empty individual template
    empty_individual = structure()
    empty_individual.position = None
    empty_individual.cost = None
    
    # Best solution ever found
    bestsol = empty_individual.deepcopy() # deepcopy garantisce che modifiche a bestsol non influenzino empty_individual e viceversa
    bestsol.cost = np.inf # imponiamo la soluzione ottima a infinito (stiamo risolvendo un problema di minimizzazione)
    
    # Initialize population
    print('Inizializzazione')
    pop = empty_individual.repeat(npop) # repeat restituisce un array di lunghezza npop di empty_individuals
    for i in range(0, npop):
        # creiamo nvar numeri distribuiti uniformemente
        # compresi fra varmin e varmax
        # i.e. inizializziamo le variabili decisionali nel nostro search space
        pop[i].position = np.random.uniform(varmin, varmax, nvar)
        pop[i].cost = costfunction(pop[i].position, X_train, X_val, y_train, y_val) # valutiamo il candidato usando la cost function
        # valuto se la nuova soluzione è migliore (più piccola) della bestsol all'iterazione precedente
        if pop[i].cost < bestsol.cost:
            bestsol = pop[i].deepcopy()
    
    print('Fine inizializzazione')
   
    # Best cost of iterations        
    # teniamo traccia del best cost alla fine di ogni iterazione
    bestcost = np.empty(maxit) 
    
#%% STEP:
    # 2- Select parents & Crossover
    # 3- Mutate offspring
    # 4- Merge main populaiton and offsprings
    # 5- Evaluate, Sort & Select
    # 6- go to step 2, if it is needed

    # Main loop
    for it in range(maxit):
        
        popc = [] # popolazione di figli (children)
        for _ in range(nc//2): # faccio //2 per essere sicuri che il numero sia intero. NB: siccome non usiamo l'indice di iterazione del for lo sostuiamo con una dummy variable "_"
            
            # Select parents randomly
            q = np.random.permutation(npop)
            p1 = pop[q[0]]
            p2 = pop[q[1]]
            
            # Perform Crossover
            c1, c2 = crossover(p1, p2, gamma)
            
            # Perform mutation
            c1 = mutate(c1, mu, sigma)
            c2 = mutate(c2, mu, sigma)
            
            # Apply bounds
            apply_bounds(c1, varmin, varmax)
            apply_bounds(c2, varmin, varmax)
            
            # Evaluate first offspring
            c1.cost = costfunction(c1.position, X_train, X_val, y_train, y_val)
            if c1.cost < bestsol.cost:
                bestsol = c1.deepcopy()
                
            # Evaluate second offspring
            c2.cost = costfunction(c2.position, X_train, X_val, y_train, y_val)
            if c2.cost < bestsol.cost:
                bestsol = c2.deepcopy()
            
            # Add offspring to popc
            popc.append(c1)
            popc.append(c2)
            
        # Merge, Sort and Select
        pop += popc # pop = pop + popc i,e, aggiungo alla popolazione la nuova polazione dei figli
        pop = sorted(pop, key=lambda x: x.cost) # key è un parametro di sorted che definisce il criterio con cui vengono ordinati i membri della popolazione. 
        # La lambda function mappa x in x.cost. Essa ritorna x.cost per ogni x contenuto nella popolazione pop. Quindi usiamo x.cost come criterio per ordinare la popolazione.
        pop = pop[0:npop] # rimuovo tutti gli elementi poco performanti tenendo solo gli npop migliori
        
        # Store best cost
        bestcost[it] = bestsol.cost
        
        #Show iteration information
        print("Iteration {}: Best Cost = {}".format(it, bestcost[it]))
        
    # Output
    out = structure()
    out.pop = pop
    out.bestsol = bestsol
    out.bestcost = bestcost
    
    return out

def crossover(p1, p2, gamma=0.1): # sfrutto l'approach "uniform crossover"
    c1 = p1.deepcopy()
    c2 = p1.deepcopy()
    # alpha è compreso in [-gamma, 1+gamma]
    alpha = np.random.uniform(-gamma, 1+gamma, *c1.position.shape) # con * andiamo a vedere ogni elemento di shape come un argomento separato per la funzione uniform
    c1.position = alpha*p1.position + (1-alpha)*p2.position
    c2.position = alpha*p2.position + (1-alpha)*p1.position
    return c1, c2
    
def mutate(x, mu, sigma): # x:soluzonie originale (non mutata), mu:mutation rate, sigma:stepsize 
    y = x.deepcopy()
    flag = (np.random.rand(*x.position.shape) <= mu) # rand ritorna valori distribuiti uniformemente fra 0 e 1
    ind = np.argwhere(flag) # abbiamo gli indici dove andremo a modificare i geni
    y.position[ind] += sigma*np.random.randn(*ind.shape) # y = y + sigma*norm -> y += sigma*norm
    return y
    
def apply_bounds(x, varmin, varmax):
    x.position = np.maximum(x.position, varmin) # Compare two arrays and returns a new array containing the element-wise maxima.
    x.position = np.minimum(x.position, varmax)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    