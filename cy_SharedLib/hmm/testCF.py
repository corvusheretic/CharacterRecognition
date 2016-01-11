import numpy as np
#from cyMartiBunke import martiBunke  # loads f.so from cc-lib: f.pyx -> f.c + fc.o -> f.so
from numpy import dtype
from numpy.random import seed, rand, randint

import pdb

from cyCalcForward import calcForward

def db_StallExec(choice):
    if(choice):
        pdb.set_trace();

def calculateForward(N,O,A,B,pi):
        scaling_factor = []
        T = len(O)
        alpha = np.zeros((T,N))
        
        #initialization
        t = 0
        alpha[t] = [pi[i] * B[i][O[t]] for i in range(N)]   
        
        def appendScalingFactor_t(t):
            sum_alpha = sum(alpha[t-1])
            if(sum_alpha==0):
                scaling_factor.append(1)
            else:
                scaling_factor.append(1.0/sum_alpha)
        
        #induction         
        for t in range(1,T):
            appendScalingFactor_t(t)
            alpha[t-1] = [scaling_factor[-1] * alpha[t-1][i] for i in range(N)]
            for j in range(N):
                prob_sum = 0  
                for i in range(N):    
                    prob_sum += alpha[t-1][i] * A[i][j]
                
                alpha[t][j] = prob_sum * B[j][O[t]]
        appendScalingFactor_t(T)
        alpha[T-1] = [scaling_factor[-1] * alpha[T-1][i] for i in range(N)]
        
        return alpha,scaling_factor

@profile
def testFunc():
    #db_StallExec(0)
    
    N = 160
    M = 150
    nLabels = 70
    
    seed(1)
    
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
    
    pi = rand(N)
    pi = (1/np.sum(pi)) * pi
    
    scaleFactor = 0.0*rand(M)
    
    #print(O)
    #print(A)
    #print(B)
    #print(pi)
    
    py_alpha,py_SF = calculateForward(N,O.tolist(),A.tolist(),B.tolist(),pi.tolist())
    print 'Data from Python Code ready.'
    #np.savetxt('data.pytxt', py_alpha, fmt='%1.10f')
    
    #==================CYTHON====================
    O     = O.astype(np.int32)
    
    sizeDefs = randint(nLabels+2, size=7)
    sizeDefs = sizeDefs.astype(np.int32)
    
    sizeDefs[0] = N   
    sizeDefs[1] = np.shape(O)[0] 
    sizeDefs[2] = np.shape(A)[0]
    sizeDefs[3] = np.shape(A)[1]
    sizeDefs[4] = np.shape(B)[0]
    sizeDefs[5] = np.shape(B)[1]
    sizeDefs[6] = np.shape(pi)[0]
    
    cy_alpha = np.zeros((M,N),dtype=np.float64)
    
    A           = A.astype(np.float64)
    B           = B.astype(np.float64)
    pi          = pi.astype(np.float64)
    scaleFactor = scaleFactor.astype(np.float64)
    
    db_StallExec(0)
    fret = calcForward( sizeDefs, O, A, B, pi, scaleFactor, cy_alpha)
    fret.apply()
    print 'Data from Cython Code ready.'
    #np.savetxt('data.cytxt', cy_alpha, fmt='%1.10f')
    xx = np.max(np.max(100.0*(np.abs(py_alpha - cy_alpha))/ np.array(py_alpha) ))
    print 'Max % Error in Apha'
    print xx
    
    db_StallExec(0)
    xx = np.max(100.0*(np.abs(np.array(py_SF) - scaleFactor) / np.array(py_SF) ))
    print 'Max % Error in SF'
    print xx

if __name__ == '__main__':
    for i in range(50):
        testFunc()
    print('Done.')