import numpy as np
from numpy import dtype
from numpy.random import seed, rand, randint
from copy import deepcopy
import pdb

from cyCalcForward import calcForward
from cyCalcBackward import calcBackward
from cyBaumWelch import fnBaumWelch

def db_StallExec(choice):
    if(choice):
        pdb.set_trace();

def baumWelch( O, A, B, pi, scaleFactor, alpha, beta):
    ''' Call with a sequence of observations, e.g. O = [0,1,0,1]. 
        The function will calculate new model parameters according 
        the baum welch formula. Will update pi.
    '''
    # Note, there is no scaling in this implementation !

    # We need to calculate the xi and gamma tables before can find the update values
    xi = np.zeros((len(O) - 1, N, N))
    gamma = np.zeros((len(O) - 1, N))
   
    # Begin with xi
    for t in range(len(O) - 1):
        s = 0
        for i in range(N):
            for j in range(N):
                xi[t][i][j] = alpha[t][i] * A[i][j] * B[j][O[t+1]] * beta[t+1][j]
                s += xi[t][i][j]
        # Normalize
        for i in range(N):
            for j in range(N):
                xi[t][i][j] *= 1/s
    
    # Now calculate the gamma table
    for t in range(len(O) - 1):
        for i in range(N):
            s = 0
            for j in range(N):
                s += xi[t][i][j]
            gamma[t][i] = s
    
    # Update model parameters
    # Update pi
    for i in range(N):
        pi[i] = gamma[0][i]

    # Update A
    #print 'Updating A'
    for i in range(N):
        for j in range(N):
            numerator = 0
            denominator = 0
            for t in range(len(O) - 1):
                numerator += xi[t][i][j]
                denominator += gamma[t][i]
            A[i][j] = numerator / denominator

    # Update B
    for j in range(N):
        for k in range(K):
            numerator = 0
            denominator = 0
            for t in range(len(O) - 1):
                if O[t] == k:
                    numerator += gamma[t][j]
                denominator += gamma[t][j]
            B[j][k] = numerator / denominator
    return xi,gamma,pi,A,B

N       = 160
M       = 150
nLabels = 70
K       = nLabels+2

@profile
def testFunc():    
    seed(0)
    
    O = randint(K, size=M)
    A = rand(N,N)
    # Normalize rows of A
    for i in range(np.shape(A)[0]):
        row  = A[i,:]
        A[i,:] = (1/np.sum(row)) * A[i,:]
    
    B = rand(N,K)
    # Normalize rows of B
    for i in range(np.shape(B)[0]):
        row  = B[i,:]
        B[i,:] = (1/np.sum(row)) * B[i,:]
    
    pi = rand(N)
    pi = (1/np.sum(pi)) * pi
    
    scaleFactor = 0.0*rand(M)
    
    #print "O:"
    #print(O)
    #print "A:"
    #print(A)
    #print "B:"
    #print(B)
    #print "scaleFactor:"
    #print(scaleFactor)
    
    #======================================================
    #alpha = calculateForward(O)
    O     = O.astype(np.int32)
    
    sizeDefs = randint(K, size=7)
    sizeDefs = sizeDefs.astype(np.int32)
    
    sizeDefs[0] = N   
    sizeDefs[1] = np.shape(O)[0] 
    sizeDefs[2] = np.shape(A)[0]
    sizeDefs[3] = np.shape(A)[1]
    sizeDefs[4] = np.shape(B)[0]
    sizeDefs[5] = np.shape(B)[1]
    sizeDefs[6] = np.shape(pi)[0]
    
    A           = A.astype(np.float64)
    B           = B.astype(np.float64)
    pi          = pi.astype(np.float64)
    scaleFactor = scaleFactor.astype(np.float64)
    
    alpha = np.zeros((M,N),dtype=np.float64)
    fretCF = calcForward( sizeDefs, O, A, B, pi, scaleFactor, alpha)
    fretCF.apply()
    
    db_StallExec(0)
    #======================================================
    #beta = calculateBackward(O)
    beta        = np.zeros((M,N),dtype=np.float64)
    scaleFactor = scaleFactor.astype(np.float64)
    A           = A.astype(np.float64)
    B           = B.astype(np.float64)
    
    fretCB = calcBackward( sizeDefs, O, A, B, scaleFactor,beta)
    fretCB.apply()
    
    py_xi, py_gamma, py_pi, py_A, py_B = baumWelch( O, 
                                                    deepcopy(A), deepcopy(B), 
                                                    deepcopy(pi), scaleFactor,
                                                    alpha, beta )
    db_StallExec(0)
    fretBW = fnBaumWelch( sizeDefs, O, A, B, pi, scaleFactor,alpha, beta )
    fretBW.apply()
    
    xx = np.max(np.max(100.0*(np.abs(py_A - A))/ np.array(py_A) ))
    print 'Max % Error in A'
    print xx
    
    xx = np.max(100.0*(np.abs(np.array(py_pi) - pi) / np.array(py_pi) ))
    print 'Max % Error in pi'
    print xx
    
    xx = np.max(np.max(100.0*(np.abs(py_B[np.where(py_B>0)] - 
                                     B[np.where(B>0)])) / 
                       py_B[np.where(py_B>0)] ))
    print 'Max % Error in B'
    print xx
    
    #==================CYTHON====================

if __name__ == '__main__':
    for i in range(10):
        testFunc()
    print('Done.')
    