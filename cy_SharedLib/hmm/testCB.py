import numpy as np
from numpy import dtype
from numpy.random import seed, rand, randint

import pdb

from cyCalcBackward import calcBackward

def db_StallExec(choice):
    if(choice):
        pdb.set_trace();

def calculateBackward(N,scaling_factor,
                      O,A,B):
    T = len(O)
    beta = np.zeros((T, N))
    #initialization
    for i in range(N):
        beta[T-1][i] = 1.0
    
    #induction
    for t in range(T-2, -1, -1):
        for i in range(N):
            prob_sum = 0
            for j in range(N):
                prob_sum += A[i][j] * (scaling_factor[t+1] *B[j][O[t+1]]) * beta[t+1][j]
            beta[t][i] = prob_sum
    return beta

@profile
def testFunc():

    N = 160
    M = 150
    nLabels = 70
    
    seed(0)
    
    O = randint(nLabels+2, size=M)
    A = rand(N,N)
    # Normalize rows of A
    for i in range(np.shape(A)[0]):
        row  = A[i,:]
        A[i,:] = (1/np.sum(row)) * A[i,:]
    
    B = rand(N,nLabels+2)
    # Normalize rows of B
    for i in range(np.shape(B)[0]):
        row  = B[i,:]
        B[i,:] = (1/np.sum(row)) * B[i,:]
    
    scaling_factor = rand(M)
    
    #print(O)
    #print(A)
    #print(B)
    #print(scaling_factor)
    
    py_beta = calculateBackward(N,scaling_factor.tolist(),
                                 O.tolist(),A.tolist(),B.tolist())
    print 'Data from Python Code ready.'
    #print(py_beta)
    #np.savetxt('data.pytxt', py_beta, fmt='%1.10f')
    
    db_StallExec(0)
    #==================CYTHON====================
    O     = O.astype(np.int32)
    
    sizeDefs = randint(nLabels+2, size=6)
    sizeDefs = sizeDefs.astype(np.int32)
    
    sizeDefs[0] = N
    sizeDefs[1] = np.shape(O)[0] 
    sizeDefs[2] = np.shape(A)[0]
    sizeDefs[3] = np.shape(A)[1]
    sizeDefs[4] = np.shape(B)[0]
    sizeDefs[5] = np.shape(B)[1]
    
    cy_beta = np.zeros((M,N),dtype=np.float64)
    
    A     = A.astype(np.float64)
    B     = B.astype(np.float64)
    scaling_factor    = scaling_factor.astype(np.float64)
    
    db_StallExec(0)
    fret = calcBackward( sizeDefs, O, A, B, scaling_factor,cy_beta)
    fret.apply()
    print 'Data from Cython Code ready.'
    #np.savetxt('data.cytxt', cy_alpha, fmt='%1.10f')
    db_StallExec(0)
    xx = np.max(np.max(100.0*(np.abs(py_beta[np.where(py_beta>0)] - 
                                     cy_beta[np.where(py_beta>0)])) / 
                       py_beta[np.where(py_beta>0)] ))
    print 'Max % Error'
    print xx

if __name__ == '__main__':
    for i in range(50):
        testFunc()
    print('Done.')
    