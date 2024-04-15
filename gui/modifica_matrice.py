# -*- coding: utf-8 -*-
"""
https://nhigham.com/2021/02/16/diagonally-perturbing-a-symmetric-matrix-to-make-it-positive-definite/

Created on Wed Apr 10 11:39:03 2024

@author: Giulio Del Corso
"""

import numpy as np

#%% Questa matrice è la d matrix che l'utente ha creato, ovviamente NON è definita positiva
M = np.eye(10).astype('float')
M[0,2] = 0.7
M[2,0] = 0.7
M[0,3] = 0.7
M[3,0] = 0.7
M[4,0] = 0.7
M[0,4] = 0.7
M[5,0] = 0.7
M[0,5] = 0.7
M_old = M



#%% Funzioni ausiliarie utili
def scale_M(M,s):
    M = s*M
    for i in range(len(M)):
        M[i,i] = 1
    return M
    
def find_s(M, epsilon = 0.04, step = 0.001):
    s = 1
    eig_val, eig_vect = np.linalg.eig(M)
    
    while min(eig_val)<epsilon:
        s = max(s-step,0)
        M_test = scale_M(M,s)
        eig_val, eig_vect = np.linalg.eig(M_test)
        
    return M_test, s



#%% Qui controlli (come già fai) che la matrice NON sia definita positiva
if not np.all(np.linalg.eigvals(M) > 0):
    print("Cavolo, questa matrice non è definita positiva!")
else:
    print("Perdiana, che fortuna, è definita positiva!")
          

          
#%% L'utente preme il tasto "Make it definite positive"
M,s = find_s(M)  # Questo normalizza tutto  



#%% Infatti se fai il check
if not np.all(np.linalg.eigvals(M) > 0):
    print("Cavolo, questa matrice non è definita positiva!")
else:
    print("Perdiana, che fortuna, è definita positiva!")
