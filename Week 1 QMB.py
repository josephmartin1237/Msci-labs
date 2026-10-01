# -*- coding: utf-8 -*-
"""
@author: Joe Martin
"""
#Libraries
import numpy as np 
import scipy.linalg as lin

#Function def to check hermicity 
def check_hermicity(matrix): 
    check = lin.ishermitian(matrix)
    if check == True: 
        result = 'Yes'
    elif check == False: 
        result = 'No'
    print(f"Is the chosen Hamiltonian Hermitian? {result} \n")




#Spin elements in x,y,z directions 
sx = 0.5*np.array([[0, 1], [1,0]])
sy = 0.5*np.array([[0, -1.0j], [1.0j, 0]])
sz = 0.5*np.array([[1, 0], [0, -1]])

#Identity
identity_2d = np.identity(2)


##########################
# Problem 2
##########################

sx_sx = np.kron(sx,sx)
sy_sy = np.kron(sy,sy)
sz_sz = np.kron(sz,sz)


Hamiltonian = np.sum((sx_sx,sy_sy,sz_sz), axis = 0)
#print(Hamiltonian)
#check_hermicity(Hamiltonian)

##########################
# Problem 3 
##########################

eigenvalues, eigenvectors = np.linalg.eigh(Hamiltonian)

print(f"The eigenvalues of the Hamiltonian are {eigenvalues}")


