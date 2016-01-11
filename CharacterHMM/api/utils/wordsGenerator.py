'''
Created on Nov 16, 2015

@author: kalyan
'''
from random import random
from random import choice
import unittest
import logging

#englishAlphabet=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
#                 'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

englishAlphabet=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

def getExampleAlphabet():
    return englishAlphabet

def generateExamplesForWord(word="dog", number_of_examples=100, poelap=0.03, poelenl=0.7, powlap=0.1, polmap=0.01, alphabet=englishAlphabet):
    '''
    Function that generate misspelled versions of a word given propabilities
    defined by the parameters.  
        
    Parameters:
    word    = the word that the examples shall be generated for
    poelap  = probability of extra letter at position
    poelenl = probability of extra letter equals neighbor letter
    powlap  = probability of wrong letter at position
    polmap  = probability of letter missing at position
    number_of_examples = the number of examples that shall be generated
    
    Returns:
    A list of size number_of_examples containing versions of the word
    '''
    #Help functions:
    def trueWithProbability(probability):
        return random() <= probability
    
    def neighborsAtPosition(word, position):
        word_length = len(word)
        if(position==0):
            return [word[0]]
        elif position < word_length:
            return [word[position-1], word[position]]
        else:
            return [word[word_length-1]]
    
    def randomSubstitution(letter):
        if(trueWithProbability(polmap)):#Letter missing at position
            return ""
        else:
            if(trueWithProbability(powlap)):#Wrong letter at position
                return choice(alphabet)
            else:
                return letter        
    
    def generateExampleForWordFromPosition(word,start_at_position=0):
        if start_at_position > len(word):
            return ""
        else:
            end = start_at_position == len(word)
            char_at_pos = "" if end else randomSubstitution(word[start_at_position])
            rest = generateExampleForWordFromPosition(word,start_at_position+1)
            if(trueWithProbability(poelap)):#probability of extra letter 
                if(trueWithProbability(poelenl)):#probability of extra letter equals to neighbor
                    neighbor = choice(neighborsAtPosition(word, start_at_position))
                    return neighbor + char_at_pos + rest
                else:
                    extra_letter = choice(alphabet)
                    return extra_letter + char_at_pos + rest
            else:
                return char_at_pos + rest
        
    #Generate the examples
    examples = []
    for i in range(number_of_examples): #@UnusedVariable
        examples.append(generateExampleForWordFromPosition(word))
    return examples


default_word_list = ["dog","cat","pig","love","hate",
                     "scala","python","summer","winter","night",
                     "daydream","nightmare","animal","happiness","sadness",
                     "tennis","feminism","fascism","socialism","capitalism"]

def generateExamplesForWords(words=default_word_list, number_of_examples=100, poelap=0.03, poelenl=0.7, powlap=0.1, polmap=0.01, alphabet=englishAlphabet):
    '''
    Generate tuples for all words in the list words of the format:
    (word, list of training examples for the words)
    
    See generateExamplesForWord for description of the rest of the parameters
    '''
    word_training_example_tuples = []
    for word in words:
        word_training_example_tuples.append((word,generateExamplesForWord(word, number_of_examples, poelap, poelenl, powlap, polmap, alphabet)))
    return word_training_example_tuples

class TestWordExampleGenerator(unittest.TestCase):

    logging.basicConfig(level=logging.INFO);
    
    def test_word_example_generator(self):
        ''' Test one word instance '''
        print(generateExamplesForWord(word="dog",number_of_examples=100))
        pass
    
    def test_words_example_generator(self):
        ''' Test multiple word instances '''
        print(generateExamplesForWords(words = default_word_list,number_of_examples=100))
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_word_']
    unittest.main()
