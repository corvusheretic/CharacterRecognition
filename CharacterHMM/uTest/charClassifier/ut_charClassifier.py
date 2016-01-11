'''
Created on Nov 20, 2015

@author: kalyan
'''
import unittest
import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
    
from api.base.charClassifier import CharacterClassifier
#from api.base.configureHMM import ConfigureHMM
from api.features.stripFeature import naiveStripFeatureExtractor

import shutil
from debug.debugger import db_StallExec

#import logging
#import numpy as np

class TestcharClassifier(unittest.TestCase):
    def test_TwoCharactersScore(self):
        #test with just two letters so A and B are copied to a
        #special dir that is deleted after the test
        cwd      = os.path.dirname(os.path.realpath(__file__));
        base_dir = os.path.join(cwd, "../../character_examples")
        test_dir = os.path.join(cwd, '../../ut_charClassifier')
        
        # Create new test directory
        try:
            os.stat(test_dir)
        except:
            os.mkdir(test_dir)
        shutil.rmtree(test_dir)
        os.mkdir(test_dir)
        
        a_dir = os.path.join(base_dir,"A")
        b_dir = os.path.join(base_dir,"B")
        
        db_StallExec(1)
        
        shutil.copytree(a_dir, os.path.join(test_dir,"A"))
        shutil.copytree(b_dir, os.path.join(test_dir,"B"))
        #shutil.copytree(b_dir.getPath(), File(test_dir,"B").getPath())
        extractor = naiveStripFeatureExtractor(nr_of_divisions=7,
                                         size_classification_factor=1.3)
        #Extract features
        training_examples, test_examples = extractor.extractTrainingAndTestingExamples(test_dir, 90, 10)
        #print("training examples", training_examples)
        #print("testing examples", test_examples)
        classifier = CharacterClassifier(training_examples,
                                         nr_of_hmms_to_try = 1,
                                         fraction_of_examples_for_test = 0.3,
                                         feature_extractor=extractor,
                                         train_with_examples=False)
        before = classifier.test(test_examples)
        #Test serialization
        classifier_string = classifier.toString()
        reborn_classifier = CharacterClassifier(from_string_string=classifier_string)
        reborn_classifier_test_result = reborn_classifier.test(test_examples)
        if(reborn_classifier_test_result==before):
            pass
        else:
            raise "Something is wrong with the test result"
        classifier.train()
        after = classifier.test(test_examples)
        print("test_with_two_characters", "before", before, "after", after)
        shutil.rmtree(test_dir)
    
    def test_TwoCharactersClassify(self):
        #test with just two letters so A and B are copied to a
        #special dir that is deleted after the test
        cwd      = os.path.dirname(os.path.realpath(__file__));
        base_dir = os.path.join(cwd, "../../character_examples")
        test_dir = os.path.join(cwd, '../../ut_charClassifier')
        
        # Create new test directory
        try:
            os.stat(test_dir)
        except:
            os.mkdir(test_dir)
        shutil.rmtree(test_dir)
        os.mkdir(test_dir)
        
        a_dir = os.path.join(base_dir,"A")
        b_dir = os.path.join(base_dir,"B")
        
        db_StallExec(1)
        
        shutil.copytree(a_dir, os.path.join(test_dir,"A"))
        shutil.copytree(b_dir, os.path.join(test_dir,"B"))
        #shutil.copytree(b_dir.getPath(), File(test_dir,"B").getPath())
        extractor = naiveStripFeatureExtractor(nr_of_divisions=7,
                                         size_classification_factor=1.3)
        #Extract features
        training_examples, test_examples = extractor.extractTrainingAndTestingExamples(test_dir, 90, 10)
        #print("training examples", training_examples)
        #print("testing examples", test_examples)
        classifier = CharacterClassifier(training_examples,
                                         nr_of_hmms_to_try = 1,
                                         fraction_of_examples_for_test = 0.3,
                                         feature_extractor=extractor,
                                         train_with_examples=False)
        before = classifier.test(test_examples)
        #Test serialization
        classifier_string = classifier.toString()
        reborn_classifier = CharacterClassifier(from_string_string=classifier_string)
        reborn_classifier_test_result = reborn_classifier.test(test_examples)
        if(reborn_classifier_test_result==before):
            pass
        else:
            raise "Something is wrong with the test result"
        classifier.train()
        after = classifier.test(test_examples)
        print("test_with_two_characters", "before", before, "after", after)
        shutil.rmtree(test_dir)
    
if __name__ == "__main__":    
    unittest.main()
    