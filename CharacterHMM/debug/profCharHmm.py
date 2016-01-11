'''
Created on Dec 4, 2015

@author: kalyan
'''

import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from api.base.charClassifier import CharacterClassifier
from debug.debugger import db_StallExec
from api.features.martibunke import martiBunkeFeatureExtractor

from debug.profileSupport import profile

#======================== martiBunkeFeature ========================
@profile            
def test_martiBunkeFeatureWithClassifier():
    #test with just two letters so A and B are copied to a
    #special dir that is deleted after the test
    cwd      = os.path.dirname(os.path.realpath(__file__));
    test_dir = os.path.join(cwd, '../ut_charClassifier')
    
    
    if(1):
        extractor = martiBunkeFeatureExtractor(nr_of_divisions=7,kMeans_k=10)
    else:
        extractor = martiBunkeFeatureExtractor(nr_of_divisions=14,kMeans_k=10)
    
    #Extract features
    training_examples, test_examples = extractor.extractTrainingAndTestingFeatures(test_dir,
                                                                                   100,
                                                                                   20,
                                                                                   test_repeat=True)
    #print("training examples", training_examples)
    #print("testing examples", test_examples)
    #sys.exit(0)
    
    db_StallExec(0)
    classifier = CharacterClassifier(training_examples,
                                     nr_of_hmms_to_try = 1,
                                     fraction_of_examples_for_test = 0,
                                     feature_extractor=extractor)
    
    db_StallExec(0)
    for example in test_examples:
        l_example = list(example)
        orig_char = l_example[0]
        l_example = l_example[1]
        cnt = 0
        n_elm = len(l_example)
        classified_char=[]
        
        for string in l_example:
            char = classifier.classifyCharacterString(string)
            classified_char.append(char)
            if(char == orig_char):
                cnt += 1
        
        print('Character '+orig_char+':')
        print('Accuracy: ' + str(cnt*100.0/n_elm))
        print(classified_char)

if __name__ == '__main__':
    test_martiBunkeFeatureWithClassifier()