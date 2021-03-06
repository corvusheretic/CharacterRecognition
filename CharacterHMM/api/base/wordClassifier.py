'''
Created on Nov 17, 2015

@author: kalyan
'''

import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from api.base.DNASeqHMM import WordHMM
from api.utils.wordsGenerator import generateExamplesForWords,\
    getExampleAlphabet
from api.base.configureHMM import ConfigureHMM
import unittest
from debug.debugger import db_StallExec

class WordClassifier(object):
    '''
    Classifies a possible misspelled word to a word
    '''


    def __init__(self,
                 words_with_examples=None,
                 nr_of_hmms_to_try=3,
                 fraction_of_examples_for_test=0.1,
                 train_with_examples=True,
                 initialisation_method=ConfigureHMM.InitMethod.count_based,
                 alphabet=getExampleAlphabet(),
                 from_string_string=None,
                 topology='bw'):
        '''
        Parameters:
        words_with_examples - is a list of tuples were the first element in the tuples
        is a string representing a word that the classifier should handle and the second
        element is a list of training examples for that word.
        nr_of_hmms_to_try - creates nr_of_hmms_to_try hmms for each word and selects the one with
        highest probability for the test examples
        fraction_of_examples_for_test -  fraction of the training examples that will be used for test
        train_with_examples - if training should be performed. Otherwise init will be done but not training
        All training examples will be used for both test and training if it is set to 0
        '''
        if from_string_string != None:
            #init from string
            #"\n\n"+ in the next row is for jython bug 1469
            words,stringified_hmms = eval("\n\n"+from_string_string)
            def destringify_hmm(hmm_string):
                return WordHMM(from_string_string=hmm_string)
            hmms = map(destringify_hmm,stringified_hmms)
            self.hmms_for_words = hmms
            self.words = words
            return
        self.words_with_examples = words_with_examples
        self.nr_of_hmms_to_try = nr_of_hmms_to_try
        self.fraction_of_examples_for_test = fraction_of_examples_for_test
        self.initialisation_method  = initialisation_method
        self.alphabet = alphabet
        self.train(train_with_examples,hmmTopology=topology)

    def train(self,train_with_examples=True,
              hmmTopology='bw'):
        self.words = []
        self.hmms_for_words = []
        for word,training_examples in self.words_with_examples:
            self.words.append(word)
            test_examples = []
            actual_training_examples = []
            if(self.fraction_of_examples_for_test == 0):
                test_examples = training_examples
                actual_training_examples = training_examples
            else:
                change_pot_at = len(training_examples)*self.fraction_of_examples_for_test
                for i in range(len(training_examples)):
                    if(i<change_pot_at):
                        test_examples.append(training_examples[i])
                    else:
                        actual_training_examples.append(training_examples[i])

            word_hmm = self.createHmmForWord(word,
                                                actual_training_examples,
                                                test_examples,
                                                self.nr_of_hmms_to_try,
                                                train_with_examples,
                                                topology=hmmTopology)
            self.hmms_for_words.append(word_hmm)


    def createHmmForWord(self,
                            word,
                            training_examples,
                            test_examples,
                            nr_of_hmms_to_try,
                            train_with_examples,
                            topology='bw'):
        #Create nr_of_hmms_to_try hmms and select the one with the best result
        results=[]
        hmms=[]
        
        #print('wordClassifier #95:')
        db_StallExec(0)
        
        for i in range(nr_of_hmms_to_try):  # @UnusedVariable
            if(self.initialisation_method==ConfigureHMM.InitMethod.count_based):
                hmm = WordHMM(len(word),
                              ConfigureHMM.InitMethod.count_based,
                              training_examples,
                              alphabet=self.alphabet)
            elif(self.initialisation_method==ConfigureHMM.InitMethod.random):
                hmm = WordHMM(len(word),
                              ConfigureHMM.InitMethod.random,
                              alphabet=self.alphabet)
            else:
                raise "Init method not supported"
            if train_with_examples:
                try:
                    hmm.trainUntilStop(training_examples, 
                                       0.0, 
                                       test_examples,
                                       hmmTopology=topology)
                except ZeroDivisionError:
                    print("Divide by zero while training")
            hmms.append(hmm)
            result = hmm.apply(test_examples)
            results.append(result)
            #print("hmm " + str(i) + " for word " + word + " result " + str(result))
        max_result = max(results)
        #print("max hmm for word " + word + " max result " + str(max_result))
        return hmms[results.index(max_result)]


    def classify(self,string):
        scores = []
        db_StallExec(0)
        for hmm in self.hmms_for_words:
            score = hmm.apply([string])
            scores.append(score)
        max_score = max(scores)
        return self.words[scores.index(max_score)]

    def test(self,test_examples):
        '''
        Parameter:
        test_examples - is a list of tuples were the first element in the tuples
        is a string representing a word that the classifier should handle and the second
        element is a list of test examples for that word.

        Returns:
        Fraction of correctly classified test examples
        '''
        correctly_classified_counter = 0.0
        wrongly__classified_counter = 0.0
        for word, examples in test_examples:
            for example in examples:
                result = self.classify(example)
                if result== word:
                    correctly_classified_counter = correctly_classified_counter + 1
                else:
                    wrongly__classified_counter = wrongly__classified_counter + 1
        total_nr_of_tests = correctly_classified_counter + wrongly__classified_counter
        score = correctly_classified_counter / total_nr_of_tests
        return score

    def toString(self):
        def hmm_to_string(hmm):
            return hmm.toString()
        stringified_hmms = map(hmm_to_string, self.hmms_for_words)
        return str((self.words,stringified_hmms))



class TestWordClassifier(unittest.TestCase):

    def test_create_classifier(self):
        words = ["pig","dog","cat","bee","ape","elk","hen","cow"]
        examples = generateExamplesForWords(words,number_of_examples=1000)
        classifier = WordClassifier(examples,
                                    nr_of_hmms_to_try = 1,
                                    fraction_of_examples_for_test = 0)
        def test_classify(word):
            print("classification of " + word + " = "+ classifier.classify(word))
        #test
        map(test_classify, words)
        test_examples = ["iig","dag","catt","bae","appe","elck","hel","row"]
        map(test_classify, test_examples)
        pass
        #["dog","cat","pig","love","hate",
        #             "scala","python","summer","winter","night",
        #             "daydream","nightmare","animal","happiness","sadness",
        #             "friendliness","feminism","fascism","socialism","capitalism"]

    def test_test_score(self):
        words = ["pig","dog","cat","bee","ape","elk","hen","cow"]
        examples = generateExamplesForWords(words,number_of_examples=200)
        classifier = WordClassifier(examples,
                                    nr_of_hmms_to_try = 1,
                                    fraction_of_examples_for_test = 0.3,
                                    train_with_examples=False)
        test_examples = generateExamplesForWords(words,number_of_examples=200)
        before = classifier.test(test_examples)
        classifier.train()
        after = classifier.test(test_examples)
        print("test_test_score", "before", before, "after", after)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_word_']
    unittest.main()


