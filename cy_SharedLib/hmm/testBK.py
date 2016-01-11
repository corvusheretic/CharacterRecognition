import numpy as np
from numpy import dtype
from numpy.random import seed, rand, randint
from copy import deepcopy
import pdb

from cyCalcForward import calcForward
from cyCalcBackward import calcBackward
from cyBakis import bakis
from cProfile import Profile

def db_StallExec(choice):
    if(choice):
        pdb.set_trace();

def baumWelchBakis( sizeDefs, O, A, B, pi, scaleFactor ):
    ''' Call with a list of sequences of observations, e.g. O = [[0,1,0], [0,1,1]].
        This is an implemenation of equations 109 and 110 in Rabiner. Will NOT update
        pi as it assumed that the model is a bakis left-to-right model.'''    

    LA_MODE = 1
    if(LA_MODE):
        sumNum_A = np.zeros((N,N),dtype=np.float64)
        sumDen_A = np.zeros((N,N),dtype=np.float64)
        sumNum_B = np.zeros((N,K),dtype=np.float64)
        sumDen_B = np.zeros((N,K),dtype=np.float64)
        
        cySumNum_A = np.zeros((N,N),dtype=np.float64)
        cySumDen_A = np.zeros((N,N),dtype=np.float64)
        cySumNum_B = np.zeros((N,K),dtype=np.float64)
        cySumDen_B = np.zeros((N,K),dtype=np.float64)
        
        sliceK = len(O)
        for k in range(sliceK):
            alpha = np.zeros((len(O[k]),N),dtype=np.float64)
            beta = np.zeros((len(O[k]),N),dtype=np.float64)
            
            #alpha.append(calculateForward(O[k]))
            matOk    = (O[k]).astype(np.int32)
            fretCF = calcForward( sizeDefs, matOk, A, B, pi, scaleFactor, alpha)
            fretCF.apply()
            
            #beta.append(calculateBackward(O[k]))
            fretCB = calcBackward( sizeDefs, matOk, A, B, scaleFactor,beta)
            fretCB.apply()
        
            P = 0
            T = len(O[k])
            
            for i in range(N):
                P += alpha[T-1][i]
            
            if( P ==0 ): # Hack to avoid division by zero later
                P = np.finfo(float).eps
        
            #print "P:"
            #print(P)
            db_StallExec(0)
            
            # Update A
            # Calculate the numerator and denominator
            #T = len(O[k])
            nS = np.zeros((N,N),dtype=np.float64)
            dS = np.zeros((N,N),dtype=np.float64)
            for t in range(T - 1):
                for i in range(N):
                    for j in range(N):
                        nS[i][j] += alpha[t][i] * B[j][O[k][t+1]] * beta[t+1][j]
                        dS[i][j] += alpha[t][i] * beta[t][i]
            sumNum_A += 1.0/P * nS
            sumDen_A += 1.0/P * dS
            
            # Update B
            # Calculate the numerator and denominator
            #T = len(O[k])
            nS = np.zeros((N,K),dtype=np.float64)
            dS = np.zeros((N,K),dtype=np.float64)
            
            for j in range(N):
                for l in range(K):
                    for t in range(T - 1):
                        if O[k][t] == l:
                            nS[j][l] += alpha[t][j] * beta[t][j]
                        dS[j][l] += alpha[t][j] * beta[t][j]
            sumNum_B += 1.0 / P * nS
            sumDen_B += 1.0 / P * dS
                        
            fretBK = bakis( sizeDefs, matOk, 
                            A, B, 
                            alpha, beta,
                            cySumNum_A, cySumDen_A,
                            cySumNum_B, cySumDen_B)
            fretBK.apply()
            
            xx = np.max(np.max(100.0*(np.abs(sumNum_A[np.where(sumNum_A>0)] - 
                                 cySumNum_A[np.where(cySumNum_A>0)])) / 
                   sumNum_A[np.where(sumNum_A>0)] ))            
            print "Max % error in SumNum_A: "
            print(xx)
            
            xx = np.max(np.max(100.0*(np.abs(sumDen_A[np.where(sumDen_A>0)] - 
                                 cySumDen_A[np.where(cySumDen_A>0)])) / 
                   sumDen_A[np.where(sumDen_A>0)] ))            
            print "Max % error in SumDen_A: "
            print(xx)
            
            xx = np.max(np.max(100.0*(np.abs(sumNum_B[np.where(sumNum_B>0)] - 
                                 cySumNum_B[np.where(cySumNum_B>0)])) / 
                   sumNum_B[np.where(sumNum_B>0)] ))            
            print "Max % error in SumNum_B: "
            print(xx)
            
            xx = np.max(np.max(100.0*(np.abs(sumDen_B[np.where(sumDen_B>0)] - 
                                 cySumDen_B[np.where(cySumDen_B>0)])) / 
                   sumDen_B[np.where(sumDen_B>0)] ))            
            print "Max % error in SumDen_B: "
            print(xx)

            db_StallExec(0)
        
        # Hack to avoid division by zero and scaling issues
        zPos  = (sumDen_A == 0.0).astype(np.int)
        nzPos = (sumDen_A != 0.0).astype(np.int)
        
        sumNum_A    = sumNum_A * nzPos
        sumDen_A += zPos 
        
        A = (A * zPos) + (A*sumNum_A) / sumDen_A
        
        for i in range(N):
            A[i] = (A[i]/np.sum(A[i])).tolist()
        
        zPos  = (sumDen_B == 0.0).astype(np.int)
        nzPos = (sumDen_B != 0.0).astype(np.int)
        
        sumNum_B    = sumNum_B * nzPos
        sumDen_B += zPos 
        
        B = (B * zPos) + sumNum_B / sumDen_B
        
        for j in range(N):
            B[j] = (B[j]/np.sum(B[j])).tolist()

    else:
        
        alpha = []
        beta = []
        P = []
        sliceK = len(O)
        
        for k in range(sliceK):
            #alpha.append(calculateForward(O[k]))
            matOk    = (O[k]).astype(np.int32)
            matAlpha = np.zeros((len(O[k]),N),dtype=np.float64)
            fretCF = calcForward( sizeDefs, matOk, A, B, pi, scaleFactor, matAlpha)
            fretCF.apply()
            alpha.append(matAlpha)
            
            #beta.append(calculateBackward(O[k]))
            matBeta = np.zeros((len(O[k]),N),dtype=np.float64)
            fretCB = calcBackward( sizeDefs, matOk, A, B, scaleFactor,matBeta)
            fretCB.apply()
            beta.append(matBeta)
        
        for k in range(sliceK):
            final_prob = 0
            T = len(O[k])
            
            for i in range(N):
                final_prob += alpha[k][T-1][i]
            
            if( final_prob ==0 ): # Hack to avoid division by zero later
                final_prob = np.finfo(float).eps
            P.append(final_prob)
        
        #print "P:"
        #print(P)
        db_StallExec(0)


        # Update A
        for i in range(N):
            for j in range(N):
                sum_numerator = 0
                sum_denominator = 0
                for k in range(sliceK):
                    # Calculate the numerator and denominator
                    T = len(O[k])
                    nS = 0
                    dS = 0
                    for t in range(T - 1):
                        nS += alpha[k][t][i] * B[j][O[k][t+1]] * beta[k][t+1][j]
                        dS += alpha[k][t][i] * beta[k][t][i]
                    sum_numerator   += 1.0 / P[k] * nS
                    sum_denominator += 1.0 / P[k] * dS
                
                # Hack to avoid division by zero and scaling issues
                if sum_denominator != 0.0:
                    A[i][j] = A[i][j] * (sum_numerator / sum_denominator)
            
            A[i] = (A[i]/np.sum(A[i])).tolist()
                
        # Update B
        for j in range(N):
            for l in range(K):
                sum_numerator = 0
                sum_denominator = 0
                for k in range(sliceK):
                    # Calculate the numerator and denominator
                    T = len(O[k])
                    nS = 0.0
                    dS = 0.0
                    for t in range(T - 1):
                        if O[k][t] == l:
                            nS += alpha[k][t][j] * beta[k][t][j]
                        dS += alpha[k][t][j] * beta[k][t][j]
                    sum_numerator   += 1.0 / P[k] * nS
                    sum_denominator += 1.0 / P[k] * dS
                
                # Hack to avoid division by zero and scaling issues
                if sum_denominator != 0.0:
                    B[j][l] = sum_numerator / sum_denominator
                
            B[j] = (B[j]/np.sum(B[j])).tolist()
        
    # We don't update Pi because we are assuming a Bakis HMM 
    # where one state will have pi[i] = 1.0
    return A,B


N       = 160
M       = 150
nLabels = 17
K       = nLabels+2
nSlices = 8

@profile
def testFunc():
    seed(0)
    
    O=[]
    for i in range(nSlices):
        O.append(randint(K, size=M))
        
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
    #print "pi:"
    #print(pi)
    
    db_StallExec(0)
    
    #======================================================
    #alpha = calculateForward(O)
    #O     = O.astype(np.int32)
    
    sizeDefs = randint(K, size=7)
    sizeDefs = sizeDefs.astype(np.int32)
    
    sizeDefs[0] = N
    sizeDefs[1] = np.shape(O[0])[0] 
    sizeDefs[2] = np.shape(A)[0]
    sizeDefs[3] = np.shape(A)[1]
    sizeDefs[4] = np.shape(B)[0]
    sizeDefs[5] = np.shape(B)[1]
    sizeDefs[6] = np.shape(pi)[0]
    
    A           = A.astype(np.float64)
    B           = B.astype(np.float64)
    pi          = pi.astype(np.float64)
    scaleFactor = scaleFactor.astype(np.float64)
    
    py_A, py_B = baumWelchBakis( sizeDefs, O, 
                                 deepcopy(A), deepcopy(B), 
                                 pi, scaleFactor )
    print "py_A:"
    print(py_A)
    print "py_B:"
    print(py_B)
    #print "A:"
    #print(A)
    #print "B:"
    #print(B)

if __name__ == '__main__':
    for i in range(10):
        testFunc()
    print('Done.')
    