'''
Created on Nov 8, 2015

@author: kalyan
'''

import sys
sys.path.append("../../api")
sys.path.append("../..")
import os
from os import walk

from random import random,seed
import cv2
import numpy as np

from api.imgProc.genericPreProc import scaleToFill, divideIntoSegments,\
    sortedCCsSizeList

import unittest
from debug.debugger import db_StallExec

class naiveStripFeatureExtractor(object):
    '''
    A class to extract a sequence of features from an image strip
    based on lengths of CCs that may be used as training observations
    for a HMM.
    '''
    
    feature_ids=['a','b','c','d','e','f','g','h','i','j']
    feature_pattern_to_id = {"LLL":"a",
                             "LLS":"b",
                             "LSS":"c",
                             "LSN":"d",
                             "LLN":"e",
                             "LNN":"f",
                             "SSS":"g",
                             "SSN":"h",
                             "SNN":"i",
                             "NNN":"j"}

    def __init__(self,
                 nr_of_divisions=7,
                 size_classification_factor=1.3):
        '''
        Parameters:
        * nr_of_divisions - Number of times to divide the image vertically
        * size_classification_factor -  A component in a segment is classified
        as small if the component size is less than "segment_width * size_classification_factor"
        and greater than zero otherwise it is classified as large. Zero size segments are
        classified as none.
        * nr_of_components_to_consider - The number of components to consider

        The 3 largest components in a segment are used to get a feature for that segment.
        There are 10 different possible features in every segment. The features are enumerated
        in the following list:

        feature id | comp. 1 | comp. 2 | comp. 3
        a          | L       | L       | L       |
        b          | L       | L       | S       |
        c          | L       | S       | S       |
        d          | L       | S       | N       |
        e          | L       | L       | N       |
        f          | L       | N       | N       |
        g          | S       | S       | S       |
        h          | S       | S       | N       |
        i          | S       | N       | N       |
        j          | N       | N       | N       |

        comp. = component
        L = large
        S = small
        N = none
        '''
        
        self.nr_of_divisions = nr_of_divisions
        self.size_classification_factor = size_classification_factor
    
    def extractFeatureString(self,buffered_image):
        if(len(buffered_image.shape)==3):
            img = cv2.cvtColor(buffered_image,cv2.COLOR_RGB2GRAY)
    
        _,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)            
        img = scaleToFill(img)
        _,scaled_image = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #db_StallExec(1)
        segments = divideIntoSegments(self.nr_of_divisions, scaled_image)
        #Get component sizes for the segments
        features_for_segments = [sortedCCsSizeList(s)
                                 for s in segments]
        #Make sure that there are 3 elements on the list for all segmensts
        def makeSizeofList3(List):
            if len(List)==3:
                return List
            elif len(List)>3:
                del List[len(List)-1]
                return makeSizeofList3(List)
            elif len(List)<3:
                List.append(0)
                return makeSizeofList3(List)
        features_for_segments = [makeSizeofList3(l)
                                 for l in features_for_segments]
        def classifyComponent(component_size, segment_width):
            if component_size >= (segment_width * self.size_classification_factor):
                return "L"
            elif component_size != 0:
                return "S"
            else:
                return "N"
        feature_string = ""
        for i in range(self.nr_of_divisions):
            segment_comp_sizes = features_for_segments[i]
            segment = segments[i]
            _,segment_width = np.shape(segment)
            segment_feature_string = ""
            for size in segment_comp_sizes:
                segment_feature_string = (segment_feature_string +
                                          classifyComponent(size, segment_width))
    
            feature_string = (feature_string +
                              self.feature_pattern_to_id[segment_feature_string])
    
        return feature_string
    
    def extractFeatureStringsForDir(self,
                                    dir_path,
                                    nr_of_training_examples=10000,
                                    nr_of_test_examples=0,
                                    test_repeat=False):
        
        _,_,images = walk(dir_path).next()
        images = sorted(images)
        
        nr_of_training_examples = min([nr_of_training_examples, len(images)])
        nr_of_images=len(images)

        test_example_indices = []
        if(test_repeat):
            seed(0)
        
        for i in range(nr_of_test_examples):
            random_value_selected = False
            random_number = 0
            while not random_value_selected:
                random_number = int(round(random()*(nr_of_images-1)))
                if not random_number in test_example_indices:
                    random_value_selected = True
            test_example_indices.append(random_number)

        test_example_indices.sort()
        test_example_indices.reverse()

        feature_strings = []
        for image in images:
            img  = cv2.imread(os.path.join(dir_path, image))
            feature_strings.append(self.extractFeatureString(img))
            
        #take out the test examples
        test_examples = []
        for i in test_example_indices:
            test_examples.append(feature_strings.pop(i))
        if len(feature_strings)>nr_of_training_examples:
            feature_strings = feature_strings[0:nr_of_training_examples]

        return (feature_strings, test_examples)

    def extractLabelExamplesTuplesForLibrary(self,library_path):
        example_dirs =  os.listdir(library_path)
        label_example_tuples = []
        for dir_name in example_dirs:
            label = dir_name
            ex_dir = os.path.join(library_path, dir_name)
            examples,_ = self.extractFeatureStringsForDir(ex_dir)
            label_example_tuples.append((label, examples))
        return label_example_tuples

    def extractTrainingAndTestingExamples(self,
                                           library_path,
                                           nr_of_training_examples=5000,
                                           nr_of_test_examples=10,
                                           test_repeat=False):
        example_dirs =  os.listdir(library_path)
        example_dirs = sorted(example_dirs)
        
        label_training_example_tuples = []
        label_test_example_tuples = []
        for dir_name in example_dirs:
            label = dir_name
            ex_dir = os.path.join(library_path, dir_name)
            training_examples,test_examples = self.extractFeatureStringsForDir(ex_dir,
                                                                          nr_of_training_examples,
                                                                          nr_of_test_examples,
                                                                          test_repeat)
            label_training_example_tuples.append((label, training_examples))
            label_test_example_tuples.append((label, test_examples))
        return (label_training_example_tuples,label_test_example_tuples)

class TestSimpleImageFeatureExtractor(unittest.TestCase):

    def getExampleImage(self):
        cwd         = os.path.dirname(os.path.realpath(__file__))
        example_dir = os.path.join(cwd, '../../character_examples')
        #example_dir = os.path.join(cwd, '../../SyntheticData')
        _,image_dir,_ = walk(example_dir).next()
        
        return (example_dir,image_dir)
    
    def test_extractFeatureString(self):
        exampleDir,imageDir = self.getExampleImage()
        d_dir = os.path.join(exampleDir, imageDir[0])
        _,_, imageList = walk(d_dir).next()
        image = cv2.imread(os.path.join(d_dir, imageList[0]))
        
        extractor = naiveStripFeatureExtractor(nr_of_divisions=5,
                                                size_classification_factor=4.3)
        feature_string = extractor.extractFeatureString(image)
        print("test_extractFeatureString")
        print(feature_string)

    def test_extractFeatureStringsForDir(self):
        extractor = naiveStripFeatureExtractor(nr_of_divisions=7,
                                            size_classification_factor=1.3)
        exampleDir,imageDir = self.getExampleImage()
        example_dir_path = os.path.join(exampleDir, imageDir[0])
        
        training_examples,test_examples = extractor.extractFeatureStringsForDir(
                                                                    example_dir_path,
                                                                    nr_of_training_examples=90,
                                                                    nr_of_test_examples=10)
        if len(training_examples)==90 and len(test_examples) == 10:
            pass
        else:
            raise "wrong number in retuned list"
        print("test_extractFeatureStringsForDir")
        print(training_examples,test_examples )
    
    def test_extractLabelExamplesTuplesForLibrary(self):
        extractor = naiveStripFeatureExtractor(nr_of_divisions=7,
                                            size_classification_factor=1.3)        
        library_path,_ = self.getExampleImage()
        
        db_StallExec(True)
        
        training_examples = extractor.extractLabelExamplesTuplesForLibrary(library_path)
        print("test_extractLabelExamplesTuplesForLibrary")
        print(training_examples)
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_word_']
    unittest.main()

