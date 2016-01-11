'''
Created on Nov 4, 2015

@author: kalyan
'''
import unittest
import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
    
from api.base.hmm import HMM
import logging
import numpy as np

class TestHMM_robotToy(unittest.TestCase):
    
    def freadLine(self,fptr):
        line=''
        while( True):
            char = fptr.read(1)
            if(char == '' or char =='\n'):
                break #EOF reached
            else:
                line = line + char
        
        return line
                
    def toIndex(self,alphabet_str):
        '''
        Returns a indices representation of the HMM o/p string
        '''
        return map(lambda x: self.V.index(x), alphabet_str)
    
    def setUp(self):
        cwd     = os.path.dirname(os.path.realpath(__file__));
        self.DataDir = os.path.join(cwd, '../data');
        
        # --- Test 01: Toy robot movement with no momentum
        self.V = ['r','g','b','y']
        self.O = []
        self.fname_momentum = os.path.join(self.DataDir,'robot_no_momemtum.data');
        
        self.pi = np.random.rand(16,1)
        self.A  = np.random.rand(16,16)
        self.B  = np.random.rand(16,4)
        
        # Normalize matrix pi 
        cumsum = np.sum(self.pi)
        self.pi = self.pi / cumsum
        # Normalize matrix A
        cumsum = np.sum(self.A, 1)
        cumsum = np.reshape(np.tile(cumsum, (1,np.shape(self.A)[1])), np.shape(self.A), 'F')
        self.A = self.A / cumsum
        # Normalize matrix B
        cumsum = np.sum(self.B, 1)
        cumsum = np.reshape(np.tile(cumsum, (1,np.shape(self.B)[1])), np.shape(self.B), 'F')
        self.B = self.B / cumsum
        
        self.h = HMM(self.pi, self.A, self.B, self.V)
        
        self.h.toString()
        
    def test_baumWelch(self):
        
        '''Train Baum-Welch HMM on 200 sequences'''
        if(os.path.isfile(self.fname_momentum)):            
            n_seq = 0
            
            f = open(self.fname_momentum,'r')
            
            # Training HMM
            print('Training HMM \n')
            while (True):
                line = self.freadLine(f)
                #print(line)
                
                if (line != ''): # EOF check
                    if (line != '.' and line != '..'):
                        self.O.append(line.split(' ')[1])
                    else:
                        #print(self.O)
                        self.O = self.toIndex(self.O)
                        
                        self.h.log.setLevel(logging.INFO)
                        self.h.baumWelch(self.O)
                        self.O = [] # Reset obs. seq. for next training set
                        n_seq += 1
                        if (line == '..'):
                            break #Start testing
                else:
                    break
            
            if(n_seq != 200): # Undefined EOF check
                print('ERROR: Training data incomplete')
            else:
                print('Testing HMM \n')
                while (True):
                    line = self.freadLine(f)
                    if (line != ''): # EOF check
                        if (line != '.'):
                            self.O.append(line.split(' ')[1])
                        else:
                            if(n_seq >= 200):
                                # Testing HMM
                                print('Output: \n')
                                self.O = self.toIndex(self.O)
                                
                                print(self.O)
                                #print(self.pi)
                                #print(self.A)
                                #print(self.B)
                                
                                self.h.log.setLevel(logging.INFO)
                                path = self.h.viterbi(self.O)
                                print('Optimal path: \n')
                                print(path)
                                
                            self.O = [] # Reset obs. seq. for next testing set
                            n_seq += 1
                    else:
                        break
            
            if(n_seq != 400): # Undefined EOF check
                print('ERROR: Testing data incomplete')
            
            f.close()
                
        else:
            print('ERROR: Unable to find file :'+str(self.fname_momentum))

    def test_baumWelchBakis(self):
        
        '''Train Baum-Welch-Bakis HMM on 200 sequences'''
        if(os.path.isfile(self.fname_momentum)):            
            n_seq = 0
            
            f = open(self.fname_momentum,'r')
            
            O_seq = []
            while (True):
                line = self.freadLine(f)
                #print(line)
                
                if (line != ''): # EOF check
                    if (line != '.' and line != '..'):
                        self.O.append(line.split(' ')[1])
                    else:
                        #print(self.O)
                        self.O = self.toIndex(self.O)
                        O_seq.append(self.O)
                        self.O = [] # Reset obs. seq. for next training set
                        n_seq += 1
                        if (line == '..'):
                            break #Start testing
                else:
                    break
            
            if(n_seq != 200): # Undefined EOF check
                print('ERROR: Training data incomplete')
            else:
                
                # Training HMM
                print('Training HMM \n')
                self.h.log.setLevel(logging.INFO)
                self.h.baumWelchBakis(O_seq)
                
                print('Testing HMM \n')
                while (True):
                    line = self.freadLine(f)
                    if (line != ''): # EOF check
                        if (line != '.'):
                            self.O.append(line.split(' ')[1])
                        else:
                            if(n_seq >= 200):
                                # Testing HMM
                                print('Output: \n')
                                self.O = self.toIndex(self.O)
                                
                                print(self.O)
                                #print(self.pi)
                                #print(self.A)
                                #print(self.B)
                                
                                self.h.log.setLevel(logging.INFO)
                                path = self.h.viterbi(self.O)
                                print('Optimal path: \n')
                                print(path)
                                
                            self.O = [] # Reset obs. seq. for next testing set
                            n_seq += 1
                    else:
                        break
            
            if(n_seq != 400): # Undefined EOF check
                print('ERROR: Testing data incomplete')
            
            f.close()
                
        else:
            print('ERROR: Unable to find file :'+str(self.fname_momentum))
                    
if __name__ == "__main__":    
    unittest.main()
