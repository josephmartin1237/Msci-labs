# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 11:23:20 2026

@author: zv24041
"""

import numpy as np 
import scipy.linalg as lin
import matplotlib.pyplot as plt 
from sympy.physics.quantum import Commutator, Operator 
from sympy import symbols

#variables




def plot_table(matrix, title):
    fig, ax = plt.subplots()
    norm = plt.Normalize(-1, 1)
    table = ax.table(cellText=matrix,
                     loc=(0,0), cellLoc="center",
                     cellColours=plt.cm.coolwarm_r(norm(matrix)))
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            cell = table[i, j]
            cell.set_height(1/len(matrix))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title) 





def create_operators(L):
    sx = 0.5*np.array([[0, 1], [1,0]])
    sy = 0.5*np.array([[0, -1.0j], [1.0j, 0]])
    sz = 0.5*np.array([[1, 0], [0, -1]])
    identity = np.identity(2)

    sx_operator= []
    sy_operator = []
    sz_operator = []
    commutator = []


    for j in range(L):
        factors_x = [identity]*L 
        factors_y = [identity]*L
        factors_z = [identity]*L
    
        factors_x[j] = sx 
        factors_y[j] = sy
        factors_z[j] = sz
    
        operator_x = factors_x[0]
        operator_y = factors_y[0]
        operator_z = factors_z[0]
    
        for factor_x in factors_x[1:]:
            operator_x = np.kron(operator_x, factor_x)
    
        for factor_y in factors_y[1:]:
            operator_y = np.kron(operator_y, factor_y)    

        for factor_z in factors_z[1:]:
            operator_z = np.kron(operator_z, factor_z)
        
        sx_operator.append(operator_x)
        sy_operator.append(operator_y)
        sz_operator.append(operator_z)
    
    return sx_operator, sy_operator, sz_operator

def commutator(s1,s2):
    return np.dot(s1,s2) - np.dot(s2,s1)
    
sx_operator, sy_operator, sz_operator = create_operators(4)  


sx1, sy1 = commutator(sx_operator[2],sy_operator[2])

def check_commutator(commutator, s3):
    
    close = np.allclose(commutator, S_i, rtol = 1e-5 , atol = 1e-5)
    
    














    
   
