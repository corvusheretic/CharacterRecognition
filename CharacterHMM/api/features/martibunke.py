'''
Created on Nov 26, 2015

@author: kalyan
'''
import sys
sys.path.append("../../api")
sys.path.append("../..")
import os
from os import walk

from copy import deepcopy
from random import random,seed
import numpy as np
import cv2
from scipy.cluster.vq import whiten, kmeans2, vq
from api.imgProc.genericPreProc import scaleToFill, divideIntoSegments
from cyMartiBunke import martiBunke

import unittest
from debug.debugger import db_StallExec
from debug.profileSupport import profile

class martiBunkeFeatureExtractor(object):
    '''
    classdocs
    '''
    feature_ids=['a','b','c','d','e','f','g','h','i','j','k','l','m',
                 'n','o','p','q','r','s','t','u','v','w','x','y','z',
                 'A','B','C','D','E','F','G','H','I','J','K','L','M',
                 'N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    
    def __init__(self,
                 nr_of_divisions=7,
                 kMeans_k=10,
                 blksize=100,
                 dataVar=0.0001):
        '''
        Constructor
        '''
        self.nr_of_divisions = nr_of_divisions
        self.otsu_threshold  = 0.0
        self.kMeans_k= kMeans_k
        
        self.batchSize      = blksize
        self.batchIndex     = 0
        self.batchCentroids = []
        self.batch_training_example_tuples = []
        self.batch_test_example_tuples = []
        self.batch_CentroidVariance = dataVar
    
    @profile
    def getMBFeatures(self, imgStrip):    
        #Normalization check
        if (np.max(imgStrip) > 1.0):
            imgStrip  = imgStrip / 255.0
        threshold = self.otsu_threshold/255.0
        #print(threshold)
        
        _, nCols = np.shape(imgStrip)
        imgStrip     = imgStrip.astype(np.float32)
        mbFeat4Strip = np.zeros((nCols,9),dtype=np.float32)
        fret = martiBunke( imgStrip, np.shape(imgStrip)[0], np.shape(imgStrip)[1],
                   mbFeat4Strip, threshold )
        fret.apply()
    
        mbFeat4Strip = np.mean(mbFeat4Strip, 0)
        return (mbFeat4Strip.tolist())
    
    @profile
    def extractMartiBunkeTuple(self,buffered_image):
        if(len(buffered_image.shape)==3):
            img = cv2.cvtColor(buffered_image,cv2.COLOR_RGB2GRAY)
    
        _,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)            
        scaled_image = scaleToFill(img,(100,100))
        self.otsu_threshold,_ = cv2.threshold(scaled_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        #db_StallExec(1)
        segments = divideIntoSegments(self.nr_of_divisions, scaled_image)
        #Get component sizes for the segments
        tuples_for_segments = [self.getMBFeatures(s)
                                 for s in segments]
        #db_StallExec(1)
        return np.array(tuples_for_segments)
        
    def extractMartiBunkeTuplesForDir(self,
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
        
        tuple_strings = []
        for image in images:
            img  = cv2.imread(os.path.join(dir_path, image))
            tuple_strings.append(self.extractMartiBunkeTuple(img))
            
        #take out the test examples
        test_examples = []
        for i in test_example_indices:
            test_examples.append(tuple_strings.pop(i))
        if len(tuple_strings)>nr_of_training_examples:
            tuple_strings = tuple_strings[0:nr_of_training_examples]
        
        return (tuple_strings, test_examples)

    def sampledMartiBunkeTuplesForDir(self,
                                    dir_path,
                                    nr_of_sampled_examples=10000,
                                    nr_of_test_examples=0,
                                    test_repeat=False):
        
        _,_,images = walk(dir_path).next()
        images = sorted(images)
        
        nr_of_images=len(images)
        nr_of_sampled_examples = min([nr_of_sampled_examples, nr_of_images])
        
        test_example_indices     = []
        training_example_indices = []
        
        if(test_repeat):
            seed(0)
        
        for i in range(nr_of_sampled_examples): # @UnusedVariable
            random_value_selected = False
            random_number = 0
            while not random_value_selected:
                random_number = int(round(random()*(nr_of_images-1)))
                if not random_number in training_example_indices:
                    random_value_selected = True
            training_example_indices.append(random_number)
        
        training_example_indices.sort()
        training_example_indices.reverse()
        tot_example_indices = deepcopy(training_example_indices)
         
        for i in range(nr_of_test_examples): # @UnusedVariable
            random_value_selected = False
            random_number = 0
            while not random_value_selected:
                random_number = int(round(random()*(nr_of_images-1)))
                if not random_number in tot_example_indices:
                    random_value_selected = True
            test_example_indices.append(random_number)
            tot_example_indices.append(random_number)
        
        test_example_indices.sort()
        test_example_indices.reverse()
        
        tuple_strings = []
        test_examples = []
        
        i = 0
        for image in images:
            if i in training_example_indices:
                img  = cv2.imread(os.path.join(dir_path, image))
                tuple_strings.append(self.extractMartiBunkeTuple(img))
            
            if i in test_example_indices:
                img  = cv2.imread(os.path.join(dir_path, image))
                test_examples.append(self.extractMartiBunkeTuple(img))
            
            i += 1
        
        return (tuple_strings, test_examples)

    def extractMartiBunkeTuplesInBatchModeForDir(self,
                                    dir_path,
                                    idx,
                                    nr_of_training_examples=10000,
                                    nr_of_test_examples=0,
                                    test_repeat=False):
        _,_,images = walk(dir_path).next()
        images = sorted(images)
        images = images[idx*self.batchSize:idx*self.batchSize+nr_of_training_examples]
        
        nr_of_training_examples = min([nr_of_training_examples, self.batchSize])

        test_example_indices = []
        if(test_repeat):
            seed(0)
            
        for i in range(nr_of_test_examples):
            random_value_selected = False
            random_number = 0
            while not random_value_selected:
                random_number = int(round(random()*(nr_of_training_examples-1)))
                if not random_number in test_example_indices:
                    random_value_selected = True
            test_example_indices.append(random_number)

        test_example_indices.sort()
        test_example_indices.reverse()
        
        tuple_strings = []
        for image in images:
            img  = cv2.imread(os.path.join(dir_path, image))
            tuple_strings.append(self.extractMartiBunkeTuple(img))
            
        #take out the test examples
        test_examples = []
        for i in test_example_indices:
            test_examples.append(tuple_strings.pop(i))
        if len(tuple_strings)>nr_of_training_examples:
            tuple_strings = tuple_strings[0:nr_of_training_examples]
        
        return (tuple_strings, test_examples)


    def extractMartiBunkeTuplesForLibrary(self,library_path):
        example_dirs =  os.listdir(library_path)
        label_example_tuples = []
        for dir_name in example_dirs:
            label = dir_name
            ex_dir = os.path.join(library_path, dir_name)
            examples,_ = self.extractMartiBunkeTuplesForDir(ex_dir)
            label_example_tuples.append((label, examples))
        return label_example_tuples

#==========================================================================================
#----------------------------------- Supporting Methods -----------------------------------

    def createObsMatrix(self,
                    tupShape,
                    nChars,
                    nChInstances,
                    label_example_tuples):
    
        obsArray = np.zeros(tupShape) - 1.0
        for i in range(nChars):
            for j in range(nChInstances):
                obsArray = np.vstack((obsArray,label_example_tuples[i][1][j]))
        
        return(obsArray[tupShape[0]:,:])
        
    def getKMeansLabels (self,
                     obsMatrix,
                     kMeans_k=10,
                     kMeans_init='points',
                     test_repeat=False,
                     normalizeData='y'):
        def mapIds(idx):
            return self.feature_ids[idx]
        
        if(normalizeData=='y'):
            obsArray = whiten(obsMatrix) # Normalize data
        else:
            obsArray = obsMatrix
        
        if( kMeans_init == 'matrix'):
            labels,centroids = vq(obsArray, kMeans_k) #centroids is distortion, its here to complete function prototype 
        else:
            if(test_repeat):
                indices = []
                seed(0)
                nr_of_obs = np.shape(obsArray)[0]
                for i in range(kMeans_k):  # @UnusedVariable
                    random_value_selected = False
                    random_number = 0
                    while not random_value_selected:
                        random_number = int(round(random()*(nr_of_obs-1)))
                        if not random_number in indices:
                            random_value_selected = True
                    indices.append(random_number)
                indices.sort()
                
                kMeans_k = [] # Populate kMeans_k as an initial guess matrix
                for i in range(len(indices)):
                    obsRow = obsArray[i,:]
                    obsRow = obsRow.tolist()
                    kMeans_k.append(obsRow)
                kMeans_k = np.array(kMeans_k)
                centroids, labels = kmeans2(obsArray, k=kMeans_k, minit='matrix')
            else:
                centroids, labels = kmeans2(obsArray, k=kMeans_k, minit=kMeans_init)
        
        #db_StallExec(1)
        labels = map(mapIds, labels.tolist())
        
        return centroids, labels 

    def extractMBFeatureString(self,
                           tupShape,
                           nChars,
                           nChInstances,
                           labels,
                           label_example_tuples):
    
        strLength = tupShape[0]
        cnt = 0
        for i in range(nChars):
            for j in range(nChInstances):
                charStr = ''.join(labels[cnt:cnt+strLength])
                label_example_tuples[i][1][j] = charStr
                cnt += strLength
        
        return(label_example_tuples)
    
    def appendBatchLabeledTuples(self,
                                 training_example_tuples,
                                 test_example_tuples):
                # Process training data
                if(len(self.batch_training_example_tuples) == 0):
                    self.batch_training_example_tuples = training_example_tuples                    
                else:
                    for label,tup in training_example_tuples:
                        for i in range(len(self.batch_training_example_tuples)):
                            if(label == self.batch_training_example_tuples[i][0]):
                                self.batch_training_example_tuples[i][1] += tup
                
                # Process testing data
                if(len(self.batch_test_example_tuples) == 0):
                    self.batch_test_example_tuples = test_example_tuples                    
                else:
                    for label,tup in test_example_tuples:
                        for i in range(len(self.batch_test_example_tuples)):
                            if(label == self.batch_test_example_tuples[i][0]):
                                self.batch_test_example_tuples[i][1] += tup

#==========================================================================================
    def extractTrainingAndTestingFeatures(self,
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
            training_examples,test_examples = self.extractMartiBunkeTuplesForDir(ex_dir,
                                                                          nr_of_training_examples,
                                                                          nr_of_test_examples,
                                                                          test_repeat)
            label_training_example_tuples.append((label, training_examples))
            label_test_example_tuples.append((label, test_examples))
            
        #Convert array of stripe-tuples to a concatenated array of observations
        
        #========================== Training Data ==========================# 
        # This is to create an dummy array to begin vertical stacking,
        # will be discarded once the final observation matrix is formed.
        
        obsMatrix= self.createObsMatrix(np.shape(training_examples[0]),
                                   len(label_training_example_tuples),
                                   len(training_examples),
                                   label_training_example_tuples)
        
        #Apply k-Means clustering on each strip
        centroids, labels = self.getKMeansLabels (obsMatrix,
                                             kMeans_k=self.kMeans_k,
                                             kMeans_init='points',
                                             test_repeat=test_repeat)
        #print labels
        
        label_training_example_tuples = self.extractMBFeatureString(np.shape(training_examples[0]),
                                                               len(label_training_example_tuples),
                                                               len(training_examples),
                                                               labels,
                                                               label_training_example_tuples)
        #========================== Test Data ==========================# 
        
        obsMatrix= self.createObsMatrix(np.shape(test_examples[0]),
                                   len(label_test_example_tuples),
                                   len(test_examples),
                                   label_test_example_tuples)
        #Apply k-Means clustering on each strip
        _, labels = self.getKMeansLabels (obsMatrix,
                                     kMeans_k=centroids,
                                     kMeans_init='matrix')
        #print labels
        #sys.exit(0)
        label_test_example_tuples = self.extractMBFeatureString(np.shape(test_examples[0]),
                                                               len(label_test_example_tuples),
                                                               len(test_examples),
                                                               labels,
                                                               label_test_example_tuples)
        return (label_training_example_tuples,label_test_example_tuples)

    def batchModeTrainingAndTestingFeatures(self,
                                           library_path,
                                           nr_of_training_examples=5000,
                                           nr_of_test_examples=10,
                                           test_repeat=False):
        example_dirs =  os.listdir(library_path)
        example_dirs = sorted(example_dirs)
        
        _,_,images = walk(os.path.join(library_path, example_dirs[0])).next()
        images = sorted(images)
        
        nr_of_images  = len(images)
        nr_of_batches = int(np.ceil(1.0 * nr_of_images / self.batchSize))
        
        Tot_training_examples = 0
        Tot_test_examples     = 0
        for nb in range(nr_of_batches):
            if(nr_of_images >= self.batchSize):
                ntraining_examples = self.batchSize
                ntest_examples     = int(float(nr_of_test_examples*self.batchSize)/nr_of_training_examples)
            else:
                ntraining_examples = nr_of_images
                ntest_examples     = 0
                
            Tot_training_examples += ntraining_examples
            Tot_test_examples     += ntest_examples
                
            nr_of_images -= self.batchSize
            
            training_example_tuples = []
            test_example_tuples = []
            for dir_name in example_dirs:
                label = dir_name
                ex_dir = os.path.join(library_path, dir_name)
                training_examples,test_examples = self.extractMartiBunkeTuplesInBatchModeForDir(ex_dir,
                                                                                                nb,
                                                                                                ntraining_examples,
                                                                                                ntest_examples,
                                                                                                test_repeat)
                training_example_tuples.append([label, training_examples])
                test_example_tuples.append([label, test_examples])
            #Convert array of stripe-tuples to a concatenated array of observations
            
            #========================== Training Data ==========================# 
            # This is to create an dummy array to begin vertical stacking,
            # will be discarded once the final observation matrix is formed.
            #db_StallExec(0)
            
            obsMatrix= self.createObsMatrix(np.shape(training_examples[0]),
                                       len(training_example_tuples),
                                       len(training_examples),
                                       training_example_tuples)
            
            #Apply k-Means clustering on each strip
            centroids,_ = self.getKMeansLabels (obsMatrix,
                                                 kMeans_k=self.kMeans_k,
                                                 kMeans_init='points',
                                                 test_repeat=test_repeat)
            #print labels
            db_StallExec(0)
            self.batchCentroids.append(centroids)
            self.appendBatchLabeledTuples(training_example_tuples, test_example_tuples)
            
        def mergeBatchCentroids():
            if(len(self.batchCentroids) > 1000):
                obsMatrix = np.zeros((len(self.batchCentroids)*self.kMeans_k,9))
                i=0
                for arr in self.batchCentroids:
                    obsMatrix[i*self.kMeans_k:(i+1)*self.kMeans_k,:] = arr
                    i+=1
            else:
                nSamples = 10
                obsMatrix = np.zeros((nSamples*len(self.batchCentroids)*self.kMeans_k,9))
                i=0
                for arr in self.batchCentroids:
                    for kCen in range(self.kMeans_k):
                        meanVec = arr[kCen,:]
                        covMat  = self.batch_CentroidVariance*np.eye(9)
                        outArr = np.random.multivariate_normal(meanVec, covMat, nSamples)
                        obsMatrix[i*nSamples:(i+1)*nSamples,:] = outArr
                        i+=1
            
            return obsMatrix
        
        Tot_training_examples -= Tot_test_examples
        
        batchObsMatrix = mergeBatchCentroids()
        mergedCentroids,_ = self.getKMeansLabels(batchObsMatrix,
                                             kMeans_k=self.kMeans_k,
                                             kMeans_init='points',
                                             test_repeat=test_repeat,
                                             normalizeData='n')
        
        print('martibunke.py# :: Centroids merged.')
        #--------------- Label training data --------------- # 
        obsMatrix= self.createObsMatrix(np.shape(training_examples[0]),
                                   len(self.batch_training_example_tuples),
                                   Tot_training_examples,
                                   self.batch_training_example_tuples)
        #Apply k-Means clustering on each strip
        _, labels = self.getKMeansLabels (obsMatrix,
                                     kMeans_k=mergedCentroids,
                                     kMeans_init='matrix')
        
        label_training_example_tuples = self.extractMBFeatureString(np.shape(training_examples[0]),
                                                               len(self.batch_training_example_tuples),
                                                               Tot_training_examples,
                                                               labels,
                                                               self.batch_training_example_tuples)
        #--------------- Label test data --------------- # 
        obsMatrix= self.createObsMatrix(np.shape(training_examples[0]),
                                   len(self.batch_test_example_tuples),
                                   Tot_test_examples,
                                   self.batch_test_example_tuples)
        #Apply k-Means clustering on each strip
        _, labels = self.getKMeansLabels (obsMatrix,
                                     kMeans_k=mergedCentroids,
                                     kMeans_init='matrix')
        #print labels
        #sys.exit(0)
        label_test_example_tuples = self.extractMBFeatureString(np.shape(training_examples[0]),
                                                               len(self.batch_test_example_tuples),
                                                               Tot_test_examples,
                                                               labels,
                                                               self.batch_test_example_tuples)
        
        print('martibunke.py# :: Labeling of training and testing data complete.')
        return (label_training_example_tuples,label_test_example_tuples)

class TestMartiBunkeFeatureExtractor(unittest.TestCase):

    def getExampleImage(self):
        cwd         = os.path.dirname(os.path.realpath(__file__))
        #example_dir = os.path.join(cwd, '../../character_examples')
        #example_dir = os.path.join(cwd, '../../SyntheticData')
        example_dir = os.path.join(cwd, '../../UniPenn')
        _,image_dir,_ = walk(example_dir).next()
        
        return (example_dir,image_dir)
    
    #def test_extractMartiBunkeTuple(self):
    #    exampleDir,imageDir = self.getExampleImage()
    #    d_dir = os.path.join(exampleDir, imageDir[0])
    #    _,_, imageList = walk(d_dir).next()
    #    image = cv2.imread(os.path.join(d_dir, imageList[0]))
    #    
    #    db_StallExec(True)
    #    extractor = martiBunkeFeatureExtractor(nr_of_divisions=5)
    #    feature_string = extractor.extractMartiBunkeTuple(image)
    #    print("test_extractMartiBunkeTuple")
    #    print(feature_string)

    #def test_extractMartiBunkeTuplesForDir(self):
    #    extractor = martiBunkeFeatureExtractor(nr_of_divisions=7)
    #    exampleDir,imageDir = self.getExampleImage()
    #    example_dir_path = os.path.join(exampleDir, imageDir[0])
    #    
    #    db_StallExec(True)
    #    training_examples,test_examples = extractor.extractMartiBunkeTuplesForDir(
    #                                                                example_dir_path,
    #                                                                nr_of_training_examples=100,
    #                                                                nr_of_test_examples=20)
    #    if len(test_examples) == 10:
    #        pass
    #    else:
    #        raise "wrong number in retuned list"
    #    print("test_extractMartiBunkeTuplesForDir")
    #    print(training_examples,test_examples )
    
    def test_extractTrainingAndTestingFeatures(self):
        extractor = martiBunkeFeatureExtractor(nr_of_divisions=7)        
        library_path,_ = self.getExampleImage()
        
        db_StallExec(0)
        
        training_examples,test_examples = extractor.extractTrainingAndTestingFeatures(library_path,
                                                                    nr_of_training_examples=100,
                                                                    nr_of_test_examples=20)
        print("test_extractTrainingAndTestingFeatures")
        print(training_examples,test_examples)
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_word_']
    unittest.main()

