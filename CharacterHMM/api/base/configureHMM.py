'''
Created on Nov 16, 2015

@author: kalyan
'''

from random import random
from hmm import HMM
import unittest

def fixedSumRandomList(number_of_elements, acc):
    def fixedSumRandomList_iter(number_of_elements_left, list_so_far):
        if(number_of_elements_left<=0):
            return list_so_far
        else:
            max_value = max(list_so_far)
            new_number1 = max_value*random()
            new_number2 = max_value - new_number1
            del list_so_far[list_so_far.index(max_value)]
            list_so_far.append(new_number1)
            list_so_far.append(new_number2)
            return fixedSumRandomList_iter(number_of_elements_left-1, list_so_far)
    
    if(number_of_elements==0):
        return []
    else:        
        return fixedSumRandomList_iter(number_of_elements-1, [acc])

def zeroPadListToSize(curList, size):
    if(len(curList)==size):
        return curList
    elif(len(curList)<size):
        curList.insert(0,0)
        return zeroPadListToSize(curList, size)
    else:
        del curList[len(curList)-1]
        return zeroPadListToSize(curList, size)
    
def zeroPadListWithSum1(size, number_of_randoms):
    rl = fixedSumRandomList(number_of_randoms,1.0)
    return zeroPadListToSize(rl,size)
    
def zerosList(number_of_zeros):
    l = []
    for i in range(number_of_zeros):  # @UnusedVariable
        l.append(0)
    return l

def fixedSumEqualElementsList(number_of_elements, acc):
    l = []
    element_value = acc / number_of_elements
    for i in range(number_of_elements): # @UnusedVariable
        l.append(element_value)
    return l



class ConfigureHMM(HMM):
    '''
    classdocs
    '''

    class InitMethod:
        random = 0
        uniform = 1
        count_based = 2

    def __init__(self, pi, A, B, V):
        super(ConfigureHMM,self).__init__(pi, A, B, V)
        
        
class TestHMM(unittest.TestCase):

    def test_zeroPadListWithSum1(self):
        r = zeroPadListWithSum1(10, 5)
        print(r)
        if((sum(r)> 0.99 ) and (sum(r)<1.01) and r[4]==0):
            pass
        else:
            raise "fail"
    
    def test_fixedSumRandomList(self):
        r = fixedSumRandomList(50, 15)
        print(r)
        if((sum(r)>0.99*15) and (sum(r)<1.01*15)):
            pass
        else:
            raise "fail"
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_word_']
    unittest.main()
