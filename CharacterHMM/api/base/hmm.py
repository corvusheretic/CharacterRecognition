'''
Created on Oct 29, 2015

@author: kalyan
'''
import sys
sys.path.append("../..")

import unittest
import logging
import random
import math
import numpy as np

from numpy.random import rand, randint
from cyCalcForward import calcForward
from cyCalcBackward import calcBackward
from cyBakis import bakis
from cyBaumWelch import fnBaumWelch

from debug.profileSupport import profile
from debug.debugger import db_StallExec

def selectRandom(massDist):
    ''' Given a discrete distribution, for example [0.2, 0.5, 0.3], select an element.
        Note that the probabilities need to sum up to 1.0
    '''
    assert(np.sum(massDist) > 0.99 and np.sum(massDist) < 1.01)
    randRoll = random.random() # in [0,1)
    s = 0
    result = 0
    for mass in massDist:
        s += mass
        if randRoll < s:
            return result
        result+=1

class HMM(object):
    '''
    classdocs
    
    Variables, the notation is the same as in the Rabiner paper.
    A  -> State Transition probability
    B  -> Observation Probability Distribution
    V  -> Vocabulary of observations
    pi -> Initial State distribution
    N  -> Number of States
    K  -> Number of symbols in vocabulary
    q  -> current state
    t  -> current (discrete) time

    '''


    def __init__(self, pi, A, B, V):
        '''
        Constructor
        
        Initalize the HMM with the supplied values
        '''
        
        self.pi = pi
        self.A  = A
        self.N  = len(A[0])
        self.B  = B
        self.V  = V
        self.K  = len(V)
        self.q  = selectRandom(self.pi)
        self.t  = 0
        self.O  = []
        self.scaling_factor = []
        
        #logging.basicConfig(level=logging.INFO)
        #self.log = logging.getLogger('HMM_log')
        #self.log.debug(' Time is ' + str(self.t) + ', Initial State is ' + str(self.q) + ', Sequence is ' + str(self.O))
        
    def generate(self):
        '''Generate a new observation based on the current state and transition to a new state.'''
        index = selectRandom(self.B[self.q])
        observation = self.V[index]
        self.O.append(observation)
        self.q = selectRandom(self.A[self.q])
        self.t += 1
        #self.log.debug(' Time is ' + str(self.t) + ', Observed ' + observation + ', New State is ' + str(self.q) + ', Sequence is ' + str(self.O))
    
    @profile
    def calculateForward(self, O):
        self.scaling_factor = []
        T = len(O)
        
        alpha = np.zeros((T,self.N))
        scaleFactor = 0.0*rand(T)
        
        sizeDefs = randint(2, size=7)
        sizeDefs = sizeDefs.astype(np.int32)
        
        sizeDefs[0] = self.N   
        sizeDefs[1] = np.shape(O)[0] 
        sizeDefs[2] = np.shape(self.A)[0]
        sizeDefs[3] = np.shape(self.A)[1]
        sizeDefs[4] = np.shape(self.B)[0]
        sizeDefs[5] = np.shape(self.B)[1]
        sizeDefs[6] = np.shape(self.pi)[0]
        
        O           = np.array(O).astype(np.int32)
        A           = np.array(self.A).astype(np.float64)
        B           = np.array(self.B).astype(np.float64)
        pi          = np.array(self.pi).astype(np.float64)
        scaleFactor = scaleFactor.astype(np.float64)
        
        db_StallExec(0)
        fret = calcForward( sizeDefs, O, A, B, pi, scaleFactor, alpha)
        fret.apply()
        
        self.scaling_factor = scaleFactor.tolist()
    
        return alpha
    
    @profile
    def calculateBackward(self, O):
        T = len(O)
        self.calculateForward(O)
        beta = np.zeros((T, self.N))
        scaling_factor = self.scaling_factor
        
        sizeDefs = randint(2, size=6)
        sizeDefs = sizeDefs.astype(np.int32)
        
        sizeDefs[0] = self.N   
        sizeDefs[1] = np.shape(O)[0] 
        sizeDefs[2] = np.shape(self.A)[0]
        sizeDefs[3] = np.shape(self.A)[1]
        sizeDefs[4] = np.shape(self.B)[0]
        sizeDefs[5] = np.shape(self.B)[1]
        
        O           = np.array(O).astype(np.int32)
        A           = np.array(self.A).astype(np.float64)
        B           = np.array(self.B).astype(np.float64)
        scaleFactor = np.array(scaling_factor).astype(np.float64)
        
        db_StallExec(0)
        fret = calcBackward( sizeDefs, O, A, B, scaleFactor, beta)
        fret.apply()
        
        return beta

    def obsProbability(self, O):
        ''''O is an observation sequence.'''
        self.calculateForward(O)
        if(0):
            def log(x):
                return math.log10(x+np.finfo(float).eps)
            try:
                log_of_probability = -sum(map(log, self.scaling_factor))
            except OverflowError:
                return 0.0
            probability = 10 ** log_of_probability
        else:
            scaling_factor = np.array(self.scaling_factor)
            #print(np.shape(scaling_factor))
            #probability    = 10 ** -np.sum(np.log10(scaling_factor+np.finfo(float).eps), 0)
            #Changed to log probability due to long HMM word sequences
            probability    = -np.sum(np.log10(scaling_factor+np.finfo(float).eps), 0)
        
        return probability

    def viterbi(self, O):
        T = len(O)
        delta = np.zeros((T, self.N))
        psi = np.zeros((T, self.N))
        #initialization
        t = 0
        for i in range(self.N):
            #delta[t][i] = self.pi[i] * self.B[i][O[t]]
            delta[t][i] = math.log10(self.pi[i]+np.finfo(float).eps) + \
                          math.log10(self.B[i][O[t]]+np.finfo(float).eps)
            psi[t][i] = 0
        # recursion
        for t in range(1, T):
            for j in range(self.N):
                acc = []
                for i in range(self.N):
                    acc.append(delta[t-1][i] + math.log10(self.A[i][j])+np.finfo(float).eps)
                delta[t][j] = max(acc) + math.log10(self.B[j][O[t]]+np.finfo(float).eps)
                psi[t][j] = acc.index(max(acc))
        # path backtracking
        last_state = delta[T-1].tolist().index(max(delta[T-1]))
        path = [last_state]
        for t in range(T-1,0,-1):
            path.append(int(psi[t][path[-1]]))
        path.reverse()
        return path
    
    @profile    
    def baumWelch(self, O):
        ''' Call with a sequence of observations, e.g. O = [0,1,0,1]. The function will
            calculate new model paramters according the baum welch formula. Will update
            pi.
        '''
        # Note, there is no scaling in this implementation !
        alpha = self.calculateForward(O)
        beta = self.calculateBackward(O)

        # We need to calculate the xi and gamma tables before can find the update values
        sizeDefs = randint(2, size=6)
        sizeDefs = sizeDefs.astype(np.int32)
        
        sizeDefs[0] = self.N   
        sizeDefs[1] = np.shape(O)[0] 
        sizeDefs[2] = np.shape(self.A)[0]
        sizeDefs[3] = np.shape(self.A)[1]
        sizeDefs[4] = np.shape(self.B)[0]
        sizeDefs[5] = np.shape(self.B)[1]
        
        O           = np.array(O).astype(np.int32)
        A           = np.array(self.A).astype(np.float64)
        B           = np.array(self.B).astype(np.float64)
        pi          = np.array(self.pi).astype(np.float64)
        scaleFactor = np.array(self.scaling_factor).astype(np.float64)
        
       
        fretBW = fnBaumWelch( sizeDefs, O, A, B, pi, scaleFactor,alpha, beta )
        fretBW.apply()
        
        self.A  = A.tolist()
        self.B  = B.tolist()
        self.pi = pi.tolist()

    @profile
    def baumWelchBakis(self, O):
        ''' Call with a list of sequences of observations, e.g. O = [[0,1,0], [0,1,1]].
            This is an implemenation of equations 109 and 110 in Rabiner. Will NOT update
            pi as it assumed that the model is a bakis left-to-right model.'''
        sizeDefs = randint(2, size=6)
        sizeDefs = sizeDefs.astype(np.int32)
        
        sizeDefs[0] = self.N   
        sizeDefs[1] = np.shape(O[0])[0] 
        sizeDefs[2] = np.shape(self.A)[0]
        sizeDefs[3] = np.shape(self.A)[1]
        sizeDefs[4] = np.shape(self.B)[0]
        sizeDefs[5] = np.shape(self.B)[1]
        
        A           = np.array(self.A).astype(np.float64)
        B           = np.array(self.B).astype(np.float64)
        
        N = self.N
        
        sumNum_A = np.zeros(np.shape(A),dtype=np.float64)
        sumDen_A = np.zeros(np.shape(A),dtype=np.float64)
        sumNum_B = np.zeros(np.shape(B),dtype=np.float64)
        sumDen_B = np.zeros(np.shape(B),dtype=np.float64)
        
        sliceK = len(O)
        for k in range(sliceK):
            alpha = self.calculateForward(O[k])
            beta  = self.calculateBackward(O[k])
            
            #alpha.append(calculateForward(O[k]))
            matOk    = np.array(O[k]).astype(np.int32)
            fretBK = bakis( sizeDefs, matOk, 
                            A, B, 
                            alpha, beta,
                            sumNum_A, sumDen_A,
                            sumNum_B, sumDen_B)
            fretBK.apply()
        
        # Hack to avoid division by zero and scaling issues
        zPos  = (sumDen_A == 0.0).astype(np.int)
        nzPos = (sumDen_A != 0.0).astype(np.int)
        
        sumNum_A    = sumNum_A * nzPos
        sumDen_A += zPos 
        
        A = (A * zPos) + (A*sumNum_A) / sumDen_A
        
        for i in range(N):
            self.A[i] = (A[i]/np.sum(A[i])).tolist()
        
        zPos  = (sumDen_B == 0.0).astype(np.int)
        nzPos = (sumDen_B != 0.0).astype(np.int)
        
        sumNum_B    = sumNum_B * nzPos
        sumDen_B += zPos 
        
        B = (B * zPos) + sumNum_B / sumDen_B
        
        for j in range(N):
            self.B[j] = (B[j]/np.sum(B[j])).tolist()

        # We don't update Pi because we are assuming a Bakis HMM 
        # where one state will have pi[i] = 1.0

    def toIndex(self,alphabet_str):
        '''
        Returns a indices representation of the HMM o/p string
        '''
        return map(lambda x: self.V.index(x), alphabet_str)
    
    def toString(self):
        '''
        Returns a string representation of the HMM that can be used
        to recreate the HMM with the from string class method
        '''
        return str((self.pi, self.A, self.B, self.V))
    
    @classmethod
    def fromString(cls, string):
        ''' Initalize the HMM with the string representation created with toString'''
        pi, A, B, V = eval(string)
        return cls(pi, A, B, V)
        

class TestHMM(unittest.TestCase):
    def setUp(self):
        # Vocabulary
        self.V = ['a', 'b']
        # initial state probabilities
        self.pi = [0.5, 0.5]
        # row index is current state, column index is new state
        # i.e. in state 0 we have 80% chance of staying in state 0 and 20% of transition to state 1
        self.A = [[0.8, 0.2],
                  [0.1, 0.9]]
        # row index is state, column index is observation
        # i.e. in state 0 we can only observe 'a' and in state 1 we can only observe 'b'
        # when the element inside B is larger than 0, there's no domain error for log,
        # what if, we have 0.0 for some of the elements,like[[1.0][0.0],[0.1][0.9]]
        self.B = [[0.9, 0.1],
                  [0.2, 0.8]]

    def test_generate(self):
        '''Create a HMM and generate 100 observations'''
        h = HMM(self.pi, self.A, self.B, self.V)
        h.log.setLevel(logging.INFO)
        for i in range(100): # @UnusedVariable
            h.generate()

    def test_forward(self):
        ''' fixme '''
        h = HMM(self.pi, self.A, self.B, self.V)
        h.log.setLevel(logging.DEBUG)
        h.calculateForward(h.toIndex(['a','b','a','b','b','a','a','a','a',
                                   'b','b','b','b','a','b','a','b','b']))

    def test_backward(self):
        '''fixme '''
        h = HMM(self.pi, self.A, self.B, self.V)
        h.log.setLevel(logging.DEBUG)
        h.calculateBackward(h.toIndex(['a','b','a','b','b','a','a','a','a',
                                    'b','b','b','b','a','b','a','b','b']))

    def test_viterbi(self):
        '''fixme'''
        h = HMM(self.pi, self.A, self.B, self.V)
        h.log.setLevel(logging.DEBUG)
        path = h.viterbi(h.toIndex(['a','b','a','b','b','a','a','a','a',
                                     'b','b','b','b','a','b','a','b','b']))
        #this is random set
        self.assertEqual(h.viterbi(h.toIndex(['a','b','a','b','b','a','a','a','a',
                                               'b','b','b','b','a','b','a','b','b'])),
                         path)

if __name__ == '__main__':
    unittest.main()

