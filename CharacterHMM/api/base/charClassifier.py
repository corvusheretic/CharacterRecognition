'''
Created on Nov 17, 2015

@author: kalyan
'''
import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from api.base.wordClassifier import WordClassifier
from api.base.configureHMM import ConfigureHMM
from api.features.stripFeature import naiveStripFeatureExtractor
from api.features.martibunke import martiBunkeFeatureExtractor

from collections import Counter
from api.utils.HTML import Table, TableCell, TableRow

import shutil
import unittest
from debug.debugger import db_StallExec

from os import walk
import numpy as np
import cv2

import cProfile
from pstats import Stats

from debug.profileSupport import profile

class CharacterClassifier(WordClassifier):
    '''
    Works as WordClassifier with some extra features for character classification
    '''


    def __init__(self,
                 characters_with_examples=None,
                 nr_of_hmms_to_try=3,
                 fraction_of_examples_for_test=0.1,
                 train_with_examples=True,
                 initialisation_method=ConfigureHMM.InitMethod.count_based,
                 feature_extractor=None,
                 from_string_string=None,
                 featExt='mb',
                 hmmTopology='bw'):
        '''
        See WordClassifier
        '''
        if from_string_string != None:
            #init from string
            #"\n\n"+ in the next row is for jython bug 1469
            if(featExt == 'ns'):
                feature_extractor_parameters,classifer_string = eval("\n\n"+from_string_string)
                nr_of_divisions,size_classification_factor = feature_extractor_parameters
                self.feature_extractor = naiveStripFeatureExtractor(nr_of_divisions,
                                                               size_classification_factor)
                self.nr_of_segments = nr_of_divisions
            
            super(CharacterClassifier,self).__init__(from_string_string=classifer_string)
            return
        #Feature extractor is voluntary but is necessary if the classifyImage
        #method shall be used
        self.feature_extractor = feature_extractor
        #Get the number of segments created by the feature extractor
        #by looking at the length of a training example
        label,examples = characters_with_examples[0]
        self.nr_of_segments = len(examples[0])
        new_characters_with_examples = []
        for label,examples in characters_with_examples:
            new_characters_with_examples.append((label*self.nr_of_segments,examples))
        
        if(featExt == 'ns'):
            alphabet=naiveStripFeatureExtractor.feature_ids
        if(featExt == 'mb'):
            alphabet=martiBunkeFeatureExtractor.feature_ids
            
        super(CharacterClassifier,self).__init__(new_characters_with_examples,
                                                 nr_of_hmms_to_try,
                                                 fraction_of_examples_for_test,
                                                 train_with_examples,
                                                 initialisation_method,
                                                 alphabet,
                                                 topology=hmmTopology)

    def classifyCharacterString(self,string):
        classification = super(CharacterClassifier, self).classify(string)
        return classification[0]

    def classifyImage(self,buffered_image):
        string = self.feature_extractor.extract_feature_string(buffered_image)
        db_StallExec(0)
        return self.classifyCharacterString(string)

    def test(self,test_examples):
        '''
        See WordClassifier.test()
        '''
        new_test_examples = []
        for label, examples in test_examples:
            new_test_examples.append((label * self.nr_of_segments, examples))
        return super(CharacterClassifier, self).test(new_test_examples)

    def toString(self):
        if self.feature_extractor == None:
            raise "feature_extractor must be given if the character classifier shall be stringified"
        else:
            feature_extractor_parameters = (self.feature_extractor.nr_of_divisions,
                                            self.feature_extractor.size_classification_factor)
        word_classifier_string = super(CharacterClassifier,self).toString()
        return str((feature_extractor_parameters,
                    word_classifier_string))


class TestCharacterClassifier(unittest.TestCase):
    
    nStrips     = 10
    nKMClasses  = 10
    hmmTopology = 'bw'
    featExt     = 'mb'
    dataSet     = 'ucc'
    '''
    def setUp(self):
        """init each test"""
        #self.testtree = SplayTree (1000000)
        self.pr = cProfile.Profile()
        self.pr.enable()
        print "\n<<<---"

    def tearDown(self):
        """finish any test"""
        p = Stats (self.pr)
        #p.strip_dirs()
        #p.sort_stats ('cumtime')
        p.print_stats ()
        print "\n--->>>"
    '''

    def getExampleImage(self):
        cwd         = os.path.dirname(os.path.realpath(__file__))
        #example_dir = os.path.join(cwd, '../../character_examples')
        #example_dir = os.path.join(cwd, '../../SyntheticData')
        example_dir = os.path.join(cwd, "../../UniPenn")
        _,image_dir,_ = walk(example_dir).next()
        
        return (example_dir,image_dir)
    
    def scaleToFill_100(self,raster):
        
        indx = np.nonzero(raster < 1)
        
        min_x = np.min(indx[0])
        max_x = np.max(indx[0])
        min_y = np.min(indx[1])
        max_y = np.max(indx[1])
        
        '''
        width,height = np.shape(raster)
        #Get extreme values from the image
        max_x = 0
        min_x = width
        max_y = 0
        min_y = height
        for x in range(0, width):
            for y in range(0,height):
                color = isInk(x,y,raster)
                if(color):
                    if x > max_x:
                        max_x = x
                    if x < min_x:
                        min_x = x
                    if y > max_y:
                        max_y = y
                    if y < min_y:
                        min_y = y
        '''
        #Cut out the part of image containing colored pixels
        sub_image = raster[min_x:max_x+1,min_y:max_y+1]
        
        #Scale the image
        resized_image = cv2.resize(sub_image,(100, 100))
        return resized_image
    
    def writeImageToDisk(self, image_path, image):
        cv2.imwrite(image_path,image)

    #def test_withScore(self):
    #    #test with just two letters so A and B are copied to a
    #    #special dir that is deleted after the test
    #    cwd      = os.path.dirname(os.path.realpath(__file__));
    #    base_dir = os.path.join(cwd, "../../character_examples")
    #    test_dir = os.path.join(cwd, '../../ut_charClassifier')
    #    
    #    # Create new test directory
    #    try:
    #        os.stat(test_dir)
    #    except:
    #        os.mkdir(test_dir)
    #    shutil.rmtree(test_dir)
    #    os.mkdir(test_dir)
    #    
    #    a_dir = os.path.join(base_dir,"A")
    #    b_dir = os.path.join(base_dir,"B")
    #    
    #    db_StallExec(0)
    #    
    #    shutil.copytree(a_dir, os.path.join(test_dir,"A"))
    #    shutil.copytree(b_dir, os.path.join(test_dir,"B"))
    #    #shutil.copytree(b_dir.getPath(), File(test_dir,"B").getPath())
    #    extractor = naiveStripFeatureExtractor(nr_of_divisions=7,
    #                                     size_classification_factor=1.3)
    #    #Extract features
    #    training_examples, test_examples = extractor.extractTrainingAndTestingExamples(test_dir, 90, 10)
    #    #print("training examples", training_examples)
    #    #print("testing examples", test_examples)
    #    classifier = CharacterClassifier(training_examples,
    #                                     nr_of_hmms_to_try = 1,
    #                                     fraction_of_examples_for_test = 0.3,
    #                                     feature_extractor=extractor,
    #                                     train_with_examples=False)
    #    before = classifier.test(test_examples)
    #    #Test serialization
    #    classifier_string = classifier.toString()
    #    reborn_classifier = CharacterClassifier(from_string_string=classifier_string)
    #    reborn_classifier_test_result = reborn_classifier.test(test_examples)
    #    if(reborn_classifier_test_result==before):
    #        pass
    #    else:
    #        raise "Something is wrong with the test result"
    #    classifier.train()
    #    after = classifier.test(test_examples)
    #    print("test_with_two_characters", "before", before, "after", after)
    #    shutil.rmtree(test_dir)
    
    '''
    def test_scaleToFill_100(self):
        cwd      = os.path.dirname(os.path.realpath(__file__));
        base_dir,imageDirs = self.getExampleImage()
        test_dir = os.path.join(cwd, '../../ut_charClassifier')
        
        # Create new test directory
        try:
            os.stat(test_dir)
        except:
            os.mkdir(test_dir)
        shutil.rmtree(test_dir)
        os.mkdir(test_dir)
        
        for d in imageDirs:
            t_dir = os.path.join(test_dir, d)
            os.mkdir(t_dir)
            
            d_dir = os.path.join(base_dir, d)
            _,_, imageList = walk(d_dir).next()
            for f in imageList:
                img = cv2.imread(os.path.join(d_dir, f))
                if(len(img.shape)==3):
                    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
                _,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)            
                img   = self.scaleToFill_100(img)
                _,image = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                #Print image to disk to test how it looks like
                self.writeImageToDisk(os.path.join(t_dir, f), image)
    '''
    def genHTMLresults(self,
                        reults_array):        
        nameHTML = 'charClassifierTest_'+str(TestCharacterClassifier.featExt)+'_'+\
                    str(TestCharacterClassifier.nStrips)+'_'+\
                    ''.join((str(TestCharacterClassifier.nKMClasses)).split('.'))+'_'+\
                    str(TestCharacterClassifier.hmmTopology)+'.html'
        
        nameTxt = 'charClassifierTest_'+str(TestCharacterClassifier.featExt)+'_'+\
                    str(TestCharacterClassifier.nStrips)+'_'+\
                    ''.join((str(TestCharacterClassifier.nKMClasses)).split('.'))+'_'+\
                    str(TestCharacterClassifier.hmmTopology)+'.txt'
        fHTML = open(nameHTML, 'w')
        fTxt  = open(nameTxt, 'w')

        tScore = Table([
                        ['-']], header_row=(' ','Avg.','character','classification','accuracy',':',' '))
    
        tAlpha = Table([
                ['A','-','-','-','-','-','-'],
                ['B','-','-','-','-','-','-'],
                ['C','-','-','-','-','-','-'],
                ['D','-','-','-','-','-','-'],
                ['E','-','-','-','-','-','-'],
                ['F','-','-','-','-','-','-'],
                ['G','-','-','-','-','-','-'],
                ['H','-','-','-','-','-','-'],
                ['I','-','-','-','-','-','-'],
                ['J','-','-','-','-','-','-'],
                ['K','-','-','-','-','-','-'],
                ['L','-','-','-','-','-','-'],
                ['M','-','-','-','-','-','-'],
                ['N','-','-','-','-','-','-'],
                ['O','-','-','-','-','-','-'],
                ['P','-','-','-','-','-','-'],
                ['Q','-','-','-','-','-','-'],
                ['R','-','-','-','-','-','-'],
                ['S','-','-','-','-','-','-'],
                ['T','-','-','-','-','-','-'],
                ['U','-','-','-','-','-','-'],
                ['V','-','-','-','-','-','-'],
                ['W','-','-','-','-','-','-'],
                ['X','-','-','-','-','-','-'],
                ['Y','-','-','-','-','-','-'],
                ['Z','-','-','-','-','-','-'],
                ['a','-','-','-','-','-','-'],
                ['b','-','-','-','-','-','-'],
                ['c','-','-','-','-','-','-'],
                ['d','-','-','-','-','-','-'],
                ['e','-','-','-','-','-','-'],
                ['f','-','-','-','-','-','-'],
                ['g','-','-','-','-','-','-'],
                ['h','-','-','-','-','-','-'],
                ['i','-','-','-','-','-','-'],
                ['j','-','-','-','-','-','-'],
                ['k','-','-','-','-','-','-'],
                ['l','-','-','-','-','-','-'],
                ['m','-','-','-','-','-','-'],
                ['n','-','-','-','-','-','-'],
                ['o','-','-','-','-','-','-'],
                ['p','-','-','-','-','-','-'],
                ['q','-','-','-','-','-','-'],
                ['r','-','-','-','-','-','-'],
                ['s','-','-','-','-','-','-'],
                ['t','-','-','-','-','-','-'],
                ['u','-','-','-','-','-','-'],
                ['v','-','-','-','-','-','-'],
                ['w','-','-','-','-','-','-'],
                ['x','-','-','-','-','-','-'],
                ['y','-','-','-','-','-','-'],
                ['z','-','-','-','-','-','-']
            ], width='100%', header_row=('Alphabet','Rank #1', 'Rank #2','Rank #3', 'Rank #4','Rank #5', 'Others'),
            col_width=('75%'))
        
        total_score = 0
        total_chars = 0
        
        i=0
        for example in reults_array:
            orig_char = example[0]
            l_example = example[1]
            n_elm = len(l_example)
            
            cnt = Counter()
            for letter in l_example:
                cnt[letter]+=1
                
            #======== HTML Formatting =========
            if(len(cnt) > 5):
                topN = 5
            else:
                topN = len(cnt)
            
            j=0    
            othersCnt=0
            for k,val in cnt.most_common(topN):
                strVal = k+'('+str(int((float(val)/n_elm + 0.00005)*10000.0)/100.0)+'%)'
                if(k == orig_char):
                    tAlpha.rows[i][j+1] = TableCell(strVal, bgcolor='green')
                    total_score += float(val)
                else:
                    tAlpha.rows[i][j+1] = TableCell(strVal)
                othersCnt += float(val)
                j +=1
            
            total_chars += n_elm
            
            if(len(cnt.keys()) >5):
                othersCnt = n_elm - othersCnt 
                strVal = '('+str(int((othersCnt/n_elm + 0.00005)*10000.0)/100.0)+'%)'
                tAlpha.rows[i][j+1] = TableCell(strVal)
            
            i +=1
            db_StallExec(0)
            #=====================================
            #======== Txt Formatting =========
            fTxt.write(orig_char+' ::[ ')
            
            topN = len(cnt)
            for k,val in cnt.most_common(topN):
                strVal = k+'('+str(int((float(val)/n_elm + 0.00005)*10000.0)/100.0)+'%)'
                fTxt.write(strVal+' ,')
            fTxt.write(' ]\n')
            #=====================================
        
        strVal = str(int((total_score/total_chars + 0.00005)*10000.0)/100.0)+'%'
        tScore.rows[0][0] = TableCell(strVal, bgcolor='yellow')
        
        fHTML.write(str(tScore) + '<p>\n')
        fHTML.write(str(tAlpha) + '<p>\n')
        fHTML.close()
        
        fTxt.write('=========================================================================\n')
        fTxt.write('Avg Character recognition rate :: '+ strVal +'\n')
        fTxt.write('=========================================================================\n')
        fTxt.close()
        
    #======================== Feature Extractor ========================
    @profile
    def test_FeatureWithClassifier(self):
        #test with just two letters so A and B are copied to a
        #special dir that is deleted after the test
        cwd      = os.path.dirname(os.path.realpath(__file__));
        
        if(self.dataSet == 'chc'):
            test_dir = os.path.join(cwd, '../../character_examples')
            nTrainingEx = 70
            nTestingEx  = 30
        if(self.dataSet == 'ucc'):
            test_dir = os.path.join(cwd, '../../ut_charClassifier')
            nTrainingEx = 100
            nTestingEx  = 20
        if(self.dataSet == 'aup'):
            test_dir = os.path.join(cwd, '../../Adv_UniPenn')
            nTrainingEx = 3000
            nTestingEx  = 600
        
        if(TestCharacterClassifier.featExt == 'ns'):
            print('charClassifier# :: Naive-Strip feature selected.')
            #db_StallExec(1)
            extractor = naiveStripFeatureExtractor(nr_of_divisions=TestCharacterClassifier.nStrips,
                                             size_classification_factor=TestCharacterClassifier.nKMClasses)
            #Extract features
            training_examples, test_examples = extractor.extractTrainingAndTestingExamples(test_dir,
                                                                                           nTrainingEx,
                                                                                           nTestingEx,
                                                                                           test_repeat=True)
            
        if(TestCharacterClassifier.featExt == 'mb'):
            print('charClassifier# :: Marti-Bunke feature selected.')
            #db_StallExec(1)
            extractor = martiBunkeFeatureExtractor(nr_of_divisions=TestCharacterClassifier.nStrips,
                                                   kMeans_k=int(TestCharacterClassifier.nKMClasses),
                                                   blksize=100)
            #Extract features
            #training_examples, test_examples = extractor.extractTrainingAndTestingFeatures(test_dir,
            #                                                                               nTrainingEx,
            #                                                                               nTestingEx,
            #                                                                               test_repeat=True)
            training_examples, test_examples = extractor.batchModeTrainingAndTestingFeatures(test_dir,
                                                                                           nTrainingEx,
                                                                                           nTestingEx,
                                                                                           test_repeat=True)
        if(TestCharacterClassifier.hmmTopology == 'bw'):
            print('charClassifier# :: Baum-Welch topology selected.')
        if(TestCharacterClassifier.hmmTopology == 'bk'):
            print('charClassifier# :: Bakis topology selected.')
        
        classifier = CharacterClassifier(training_examples,
                                         nr_of_hmms_to_try = 1,
                                         fraction_of_examples_for_test = 0,
                                         feature_extractor=extractor,
                                         featExt=TestCharacterClassifier.featExt,
                                         hmmTopology=TestCharacterClassifier.hmmTopology)
        
        db_StallExec(0)
                
        resultsArray = []
        for example in test_examples:
            l_example = list(example)
            orig_char = l_example[0]
            l_example = l_example[1]
            
            classified_char=[]
            
            for string in l_example:
                char = classifier.classifyCharacterString(string)
                classified_char.append(char)
                
            resultsArray.append((orig_char,classified_char))
        
        self.genHTMLresults(resultsArray)
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_word_']
    if len(sys.argv) > 1:
        TestCharacterClassifier.dataSet     = sys.argv.pop()
        TestCharacterClassifier.hmmTopology = sys.argv.pop()
        TestCharacterClassifier.nKMClasses  = float(sys.argv.pop())
        TestCharacterClassifier.nStrips     = int(sys.argv.pop())
        TestCharacterClassifier.featExt     = sys.argv.pop()
    
    #print(str(TestCharacterClassifier.nKMClasses ))
    #print(str(TestCharacterClassifier.nStrips    ))
    #print(TestCharacterClassifier.hmmTopology)
    #print(TestCharacterClassifier.featExt    )
    
    unittest.main()
