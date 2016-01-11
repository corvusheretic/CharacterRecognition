'''
Created on Nov 16, 2015

@author: kalyan
'''
import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from configureHMM import ConfigureHMM
from configureHMM import zerosList
from configureHMM import fixedSumRandomList
from configureHMM import zeroPadListWithSum1
from configureHMM import fixedSumEqualElementsList
from api.utils.wordsGenerator import getExampleAlphabet
from api.utils.wordsGenerator import generateExamplesForWord

import unittest
from debug.debugger import db_StallExec

class WordHMM(ConfigureHMM):
    '''
    A HMM that represent a sequence of letter that form a word.
    It is implemented in the way described in the paper:
    Initial model selection for the Baum-Welch algorithm as applied to 
    HMMs of DNA sequences.
    '''

    def __init__(self, 
                 word_length=7, 
                 init_method=ConfigureHMM.InitMethod.random, 
                 training_examples=[],
                 alphabet=getExampleAlphabet(),
                 from_string_string = None):
        '''
        Training examples is only used if InitMethod.count_based is used
        '''
        if from_string_string != None:
            #init from string
            pi, A, B, V = eval(from_string_string)
            super(WordHMM,self).__init__(pi, A, B, V)
            return
        self.word_length = word_length
        self.init_method = init_method
        self.training_examples = training_examples
        self.alphabet = alphabet
        if(self.init_method==ConfigureHMM.InitMethod.count_based and
           len(self.training_examples)==0):
            raise "Training examples needs to be provided when init method is count based"
        
        #Construct the state transition matrix
        self.number_of_states = word_length + 2
        #state transition matrix
        A = []
        #From state 1 to state 2 the probability is 
        state1 = zerosList(self.number_of_states)
        state1[1]=1
        A.append(state1)
        for i in range(1,self.number_of_states-1):
            state_row = self.initTransitionMatrixRow(i)
            A.append(state_row)
        #last state can only be transfered to state1 with probability 1
        last_state = zerosList(self.number_of_states)
        last_state[0]=1
        A.append(last_state)
        
        #init state emission probabilities...
        self.number_of_emissions = len(self.alphabet) + 2
        B = []
        #init the first row with specific probability for @
        B.append(zerosList(self.number_of_emissions))
        B[0][0] = 1
        #init the rest emission probabilities without the last row
        for i in range(1, self.number_of_states-1):
            B.append(self.initEmissionProbablityMatrixRow(i))
        #init the last row for specific probability for $
        B.append(zerosList(self.number_of_emissions))
        B[self.number_of_states-1][self.number_of_emissions-1] = 1
        
        #Set of emission symbols
        V = ['@'] + self.alphabet + ['$']
        #Initial state
        pi = zerosList(self.number_of_states)
        pi[0] = 1
        super(WordHMM,self).__init__(pi, A, B, V)

    def initTransitionMatrixRow(self, row_index):
        if(self.init_method==ConfigureHMM.InitMethod.random):
            return zeroPadListWithSum1(self.number_of_states, self.number_of_states-row_index)
        elif(self.init_method==ConfigureHMM.InitMethod.count_based):
            row = (zerosList(row_index) + 
                   fixedSumEqualElementsList(self.number_of_states-row_index-1,0.2))
            row.insert(row_index+1,0.8)
            return row            
        else:
            raise "Init Method Not Supported"
    
    def initEmissionProbablityMatrixRow(self, row_index):
        if(self.init_method==ConfigureHMM.InitMethod.random):
            row = [0] + fixedSumRandomList(self.number_of_emissions-2, 1) + [0]
            return row
        elif(self.init_method==ConfigureHMM.InitMethod.count_based):
            nr_of_training_examples = len(self.training_examples)
            alphabet = self.alphabet
            alphabet_size = len(alphabet)
            def countPosition(position):
                #pseudocount
                use_pseudocount = True
                uniform_pseudocount = False
                if use_pseudocount:
                    if uniform_pseudocount:
                        count_list = zerosList(alphabet_size)
                        for i in range(0,alphabet_size):
                            count_list[i]= (nr_of_training_examples*0.1)/alphabet_size
                    else:
                        count_list = fixedSumRandomList(alphabet_size,
                                                          nr_of_training_examples*0.1)
                else:
                    count_list = zerosList(alphabet_size)
                #Do the counting
                for e in self.training_examples:
                    if position < len(e):
                        character_index = alphabet.index(e[position])
                        count_list[character_index] = count_list[character_index] + 1
                return count_list
            count_list = countPosition(row_index-1)
            total_count = sum(count_list)
            def normalizeElement(element):
                return element / total_count
            row = map(normalizeElement, count_list)
            return [0] + row + [0]
        else:
            raise "Init Method Not Supported"
    
    def observationFromWord(self,word):
        word_with_special_start_and_end = "@" +  word +  "$"
        observation_list = []
        for letter in word_with_special_start_and_end:
            observation_list.append(self.V.index(letter))
        return observation_list
    
    
    def trainBaumWelch(self, training_examples):
        observation_list = []
        for word in training_examples:
            observation_list = observation_list + self.observationFromWord(word)
        self.baumWelch(observation_list)
        
    def trainBaumWelchBakis(self, training_examples):
        '''bakis does not seem to work, see autotest'''
        observation_list = []
        for word in training_examples:
            observation_list.append(self.observationFromWord(word))
        self.baumWelchBakis(observation_list)
        
    def trainUntilStop(self, 
                       training_examples, 
                       delta = 0.0003, 
                       test_examples=None,
                       max_nr_of_iterations=1500,
                       hmmTopology='bw'):
        ''' Train the model using Baum Welch until stop condition is met.
            stop condition improvement < delta
           
            Parameters:
            training_examples - the example words to train with
            delta - see stop condition
            test_examples the examples used to test for improvement the training examples are used if
            set to default None'''
        actual_test_examples = []
        if test_examples==None:
            actual_test_examples = training_examples
        else:
            actual_test_examples = test_examples
        
        score = 0
        old_score = -1
        improvement = score - old_score
        iteration = 0 
        while improvement > delta and iteration < max_nr_of_iterations:
            iteration = iteration + 1
            
            # Choose HMM Topology
            if(hmmTopology=='bw'):
                #print('DNASeq#_185 :: Baum-Welch topology selected.')
                #db_StallExec(1)
                self.trainBaumWelch(training_examples)
            if(hmmTopology=='bk'):
                #print('DNASeq#_189 :: Bakis topology selected.')
                #db_StallExec(1)
                self.trainBaumWelchBakis(training_examples)
                
            old_score = score
            score = self.apply(actual_test_examples)
            improvement = score - old_score
            
            #print(improvement)
    
    def apply(self, word_list):
        '''Returns the likelihood of the word given the model'''
        probabilities_for_words = []
        for word in word_list:
            O = self.observationFromWord(word)
            #alpha_matrix = self.calc_forward(O)
            #last_row = alpha_matrix[len(alpha_matrix)-1]
            probabilities_for_words.append(self.obsProbability(O))
        average = sum(probabilities_for_words)/len(probabilities_for_words)
        return average

class TestHMM(unittest.TestCase):
    
    
    def test_with_word(self):
        word_hmm = WordHMM(word_length=3)
        if len(word_hmm.A) == 5:
            pass
        else:
            raise "The size of A is incorrect"

    def trainUntilStopConditionReached(self, word_hmm):
        examples = generateExamplesForWord(word="dog", number_of_examples=500)
        test_examples = generateExamplesForWord(word="dog", number_of_examples=40)
        before = word_hmm.apply(test_examples)
        word_hmm.trainUntilStop(examples, delta = 0.0, test_examples = test_examples)
        after = word_hmm.apply(test_examples)
        if(after > before):
            print("test_train_until_stop_condition_reached", "before", before, "after", after)
            pass
        else:
            raise "The training does not seem to work good before " + str(before) + " after " + str(after)

    def test_train_until_stop_condition_reached(self):
        print("random init")
        self.trainUntilStopConditionReached(WordHMM(word_length=3))
        print("count based init")
        init_training_examples = generateExamplesForWord(word="dog", number_of_examples=40)
        self.trainUntilStopConditionReached(WordHMM(3, 
                                                    ConfigureHMM.InitMethod.count_based,
                                                    init_training_examples))

    def test_train_with_stop_condition_bakis(self):
        word_hmm = WordHMM(word_length=3)
        examples = generateExamplesForWord(word="dog", number_of_examples=1000)
        test_examples = generateExamplesForWord(word="dog", number_of_examples=10)
        score = 0
        old_score = -1
        print("bakis")
        while score > old_score:
            old_score = score
            
            db_StallExec(False)
            
            word_hmm.trainBaumWelchBakis(examples)
            score = word_hmm.apply(test_examples)
            print("score " + str(score))
        print("final score " + str(score))

    def test_train(self):
        word_hmm = WordHMM(word_length=3)
        examples = generateExamplesForWord(word="dog", number_of_examples=1000)
        test_examples = generateExamplesForWord(word="dog", number_of_examples=10)
        other_test_examples = generateExamplesForWord(word="pig", number_of_examples=10)
        before = word_hmm.apply(test_examples)
        word_hmm.trainBaumWelch(examples)
        after = word_hmm.apply(test_examples)
        other_test_examples_test = word_hmm.apply(other_test_examples)
        if(after > before and other_test_examples_test < after):
            print("test train", "before", before, "after", after)
            pass
        else:
            raise "The training does not seem to work good"
            
        print(["before", before, "after", after,"other_test_examples_test", other_test_examples_test])
        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_word_']
    unittest.main()
    