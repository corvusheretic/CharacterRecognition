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
        
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('HMM_log')    
        self.log.debug(' Time is ' + str(self.t) + ', Initial State is ' + str(self.q) + ', Sequence is ' + str(self.O))
        
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
        
        f_hand = file("calcForward.dat",'a')
        f_hand.write(str(self.N))
        f_hand.write("\n\n")
        
        O_mat = np.array(O)
        np.savetxt(f_hand, O_mat, fmt='%3.1f',delimiter=',')
        f_hand.write("\n")
        
        A_mat = np.array(self.A)
        np.savetxt(f_hand, A_mat, fmt='%2.8f',delimiter=',')
        f_hand.write("\n")
        
        B_mat = np.array(self.B)
        np.savetxt(f_hand, B_mat, fmt='%2.8f',delimiter=',')
        f_hand.write("\n")
        
        pi_mat = np.array(self.pi)
        np.savetxt(f_hand, pi_mat, fmt='%2.8f',delimiter=',')
        f_hand.write("========== Data Done==========\n")
        f_hand.close()
        
        #db_StallExec(1)
        
        #initialization
        t = 0
        alpha[t] = [self.pi[i] * self.B[i][O[t]] for i in range(self.N)]   
        
        def appendScalingFactor_t(t):
            sum_alpha = sum(alpha[t-1])
            if(sum_alpha==0):
                self.scaling_factor.append(1)
            else:
                self.scaling_factor.append(1.0/sum_alpha)
        
        #induction         
        for t in range(1,T):
            appendScalingFactor_t(t)
            alpha[t-1] = [self.scaling_factor[-1] * alpha[t-1][i] for i in range(self.N)]
            for j in range(self.N):
                prob_sum = 0  
                for i in range(self.N):    
                    prob_sum += alpha[t-1][i] * self.A[i][j]
                #self.log.debug('t is ' + str(t) + ', i = ' + str(i) + ', j = ' +str(j) + ', O[t] = ' + str(O[t]) + ', prob_sum = ' + str(prob_sum) + ', B[j][O[t]] = ' + str(self.B[j][O[t]]))
                alpha[t][j] = prob_sum * self.B[j][O[t]]
        appendScalingFactor_t(T)
        alpha[T-1] = [self.scaling_factor[-1] * alpha[T-1][i] for i in range(self.N)]
        
        #f_hand = file("alpha_mat.dat",'a')
        #alpha_mat = np.array(alpha)
        #np.savetxt(f_hand, alpha_mat, fmt='%2.8e',delimiter=',')
        #f_hand.write("\n\n")
        #f_hand.close()
        
        return alpha
    
    @profile
    def calculateBackward(self, O):
        T = len(O)
        self.calculateForward(O)
        beta = np.zeros((T, self.N))
        scaling_factor = self.scaling_factor
        #initialization
        for i in range(self.N):
            beta[T-1][i] = 1.0
        #self.log.debug(' beta is ' + str(beta))
        
        #induction
        for t in range(T-2, -1, -1):
            for i in range(self.N):
                prob_sum = 0
                for j in range(self.N):
                    #print('scaling_factor=' + str(scaling_factor))
                    prob_sum += self.A[i][j] * (scaling_factor[t+1] *self.B[j][O[t+1]]) * beta[t+1][j]
                beta[t][i] = prob_sum
        #self.log.debug(' beta is ' + str(beta))
        return beta

    def obsProbability(self, O):
        ''''O is an observation sequence.'''
        self.calculateForward(O)
        def log(x):
            return math.log10(x+np.finfo(float).eps)
        try:
            log_of_probability = -sum(map(log, self.scaling_factor))
        except OverflowError:
            return 0.0
        probability = 10 ** log_of_probability
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
        xi = np.zeros((len(O) - 1, self.N, self.N))
        gamma = np.zeros((len(O) - 1, self.N))
       
        # Begin with xi
        for t in range(len(O) - 1):
            s = 0
            for i in range(self.N):
                for j in range(self.N):
                    #self.log.debug(' t = ' + str(t) + ', i = ' + str(i) + ', j = ' +str(j))
                    xi[t][i][j] = alpha[t][i] * self.A[i][j] * self.B[j][O[t+1]] * beta[t+1][j]
                    s += xi[t][i][j]
            # Normalize
            for i in range(self.N):
                for j in range(self.N):
                    xi[t][i][j] *= 1/s

        # Now calculate the gamma table
        for t in range(len(O) - 1):
            for i in range(self.N):
                s = 0
                for j in range(self.N):
                    s += xi[t][i][j]
                gamma[t][i] = s
        # Update model parameters
        # Update pi
        for i in range(self.N):
            self.pi[i] = gamma[0][i]
        # Update A
        #print 'Updating A'
        for i in range(self.N):
            for j in range(self.N):
                numerator = 0
                denominator = 0
                for t in range(len(O) - 1):
                    numerator += xi[t][i][j]
                    denominator += gamma[t][i]
                self.A[i][j] = numerator / denominator
        # Update B
        for j in range(self.N):
            for k in range(self.K):
                numerator = 0
                denominator = 0
                for t in range(len(O) - 1):
                    if O[t] == k:
                        numerator += gamma[t][j]
                    denominator += gamma[t][j]
                self.B[j][k] = numerator / denominator

    @profile
    def baumWelchBakis(self, O):
        ''' Call with a list of sequences of observations, e.g. O = [[0,1,0], [0,1,1]].
            This is an implemenation of equations 109 and 110 in Rabiner. Will NOT update
            pi as it assumed that the model is a bakis left-to-right model.'''
        alpha = []
        beta = []
        P = []
        K = len(O)
        
        for k in range(K):
            alpha.append(self.calculateForward(O[k]))
            beta.append(self.calculateBackward(O[k]))
            final_prob = 0
            T = len(O[k])
            
            for i in range(self.N):
                final_prob += alpha[k][T-1][i]
            P.append(final_prob)
        
        # Update A
        for i in range(self.N):
            for j in range(self.N):
                sum_numerator = 0
                sum_denominator = 0
                for k in range(K):
                    # Calculate the numerator
                    T = len(O[k])
                    s = 0
                    for t in range(T - 1):
                        s += alpha[k][t][i] * self.A[i][j] * self.B[j][O[k][t+1]] * beta[k][t+1][j]
                    sum_numerator += 1.0 / P[k] * s
                    # Calculate the denominator
                    s = 0
                    for t in range(T - 1):
                        s += alpha[k][t][i] * beta[k][t][i]
                    sum_denominator += 1.0 / P[k] * s
                
                # Hack to avoid division by zero and scaling issues
                if sum_denominator != 0.0:
                    self.A[i][j] = sum_numerator / sum_denominator
            
            self.A[i] = (self.A[i]/np.sum(self.A[i])).tolist()
            

        # Update B
        for j in range(self.N):
            for l in range(self.K):
                sum_numerator = 0
                sum_denominator = 0
                for k in range(K):
                    # Calculate the numerator
                    T = len(O[k])
                    s = 0.0
                    for t in range(T - 1):
                        if O[k][t] == l:
                            s += alpha[k][t][j] * beta[k][t][j]
                    sum_numerator += 1.0 / P[k] * s
                    # Calculate the denominator
                    s = 0.0
                    for t in range(T - 1):
                        s += alpha[k][t][j] * beta[k][t][j]
                    sum_denominator += 1.0 / P[k] * s
                
                # Hack to avoid division by zero and scaling issues
                if sum_denominator != 0.0:
                    self.B[j][l] = sum_numerator / sum_denominator
                
            self.B[j] = (self.B[j]/np.sum(self.B[j])).tolist()

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

