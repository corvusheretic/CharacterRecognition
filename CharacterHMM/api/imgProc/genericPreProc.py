'''
Created on Nov 8, 2015

@author: kalyan
'''

import sys
import os
import shutil
from os import walk
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import cv2
import numpy as np
from sets import Set

import unittest
from debug.debugger import db_StallExec

def isInk(x,y, raster):
    '''Returns true if pixel has black color'''
    pixel = raster[x,y]
    if(pixel == 0):
        return True
    else:
        return False
    
def scaleToFill(raster,
                (width,height) = (0,0)):
    
    if ((width,height) == (0,0)):
        width,height = np.shape(raster)
    #Get extreme values from the image
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
    resized_image = cv2.resize(sub_image,(height, width))
    return resized_image

def divideIntoSegments(nr_of_segments, image):
    _,width = np.shape(image)
    segment_width = width / nr_of_segments
    
    def createSegment(start_pos):
        end = start_pos + segment_width
        if end > width:
            this_segment_width = segment_width - (end-width)
        #elif (width - end - segment_width) < 0:
        #    this_segment_with = width - start_pos
        else:
            this_segment_width = segment_width
        seg = image[:,start_pos:start_pos+this_segment_width]
        return seg
    
    segment_starts = range(0,width, segment_width)
    if len(segment_starts) > nr_of_segments:
        del segment_starts[len(segment_starts)-1]
    segments = [createSegment(s) for s in segment_starts]
    
    return segments

def extractSortedComponentSizeList(image):
    #Search for unprocessed colored pixels and find the component
    width,height = np.shape(image)
    #make sure we don't run out of stack space
    old_rec_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(width*height)
    #Remember which pixels have been processed
    processed_colored_pixels = Set()
    def neighbourPixels(pixel):
        x,y = pixel
        neighbours = [(x-1,y-1),
                      (x-1,y),
                      (x-1,y+1),
                      (x,y-1),
                      (x,y+1),
                      (x+1,y-1),
                      (x+1,y),
                      (x+1,y+1)]
        valid_neighbours = [(x,y) for (x,y) in neighbours 
                            if(x >= 0 and x < width and
                               y >= 0 and y < height)]
        return valid_neighbours
        
    def findComponentLength(start_pixel):
        x,y = start_pixel
        if not isInk(x, y, image):
            return 0
        elif start_pixel in processed_colored_pixels:
            return 0
        else:
            processed_colored_pixels.add(start_pixel)
            neighbours = neighbourPixels(start_pixel)
            
            lengths_of_neighbour_components = [findComponentLength(p)
                                              for p in neighbours]
            return 1 + sum(lengths_of_neighbour_components)
    component_lengths = [length for length in 
                         [findComponentLength((x,y)) for x in range(width) for y in range(height)]
                         if(length>0)]
    #Set stack limit back to normal    
    sys.setrecursionlimit(old_rec_limit)
    #Component lengths shall be sorted with the largest first
    component_lengths.sort()
    component_lengths.reverse()
    return component_lengths

def sortedCCsSizeList(image):
    _,contours,_ = cv2.findContours(255-image,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)
    component_lengths=[]
    
    db_StallExec(False)
    
    for comp in contours:
        length = cv2.arcLength(comp,closed=False)
        component_lengths.append(length)
    
    component_lengths.sort()
    component_lengths.reverse()
    return component_lengths
    
class TestImagePreprocessor(unittest.TestCase):
    
    def getExampleImage(self):
        cwd         = os.path.dirname(os.path.realpath(__file__))
        #example_dir = os.path.join(cwd, '../../character_examples')
        example_dir = os.path.join(cwd, '../../SyntheticData')
        _,image_dir,_ = walk(example_dir).next()
        
        return (example_dir,image_dir)
    
    def writeImageToDisk(self, image_path, image):
        cv2.imwrite(image_path,image)
    
    def test_scaleToFill(self):
        '''
        Unit-test for scaleToFill
        '''
        
        cwd = os.path.dirname(os.path.realpath(__file__));
        tmp_dir = os.path.join(cwd, 'ut_scaleToFill');
        
        # Create new temp directory for each test
        try:
            os.stat(tmp_dir);
        except:
            os.mkdir(tmp_dir);
        shutil.rmtree(tmp_dir)
        os.mkdir(tmp_dir)
        
        exampleDir,imageDir = self.getExampleImage()
        for d in imageDir:
            t_dir = os.path.join(tmp_dir, d)
            os.mkdir(t_dir)
            
            d_dir = os.path.join(exampleDir, d)
            _,_, imageList = walk(d_dir).next()
            for f in imageList:
                img = cv2.imread(os.path.join(d_dir, f))
                if(len(img.shape)==3):
                    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
                _,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)            
                img = scaleToFill(img)
                _,image = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                #Print image to disk to test how it looks like
                self.writeImageToDisk(os.path.join(t_dir, f), image)
    
    def test_divideIntoSegments(self):
        '''
        Unit-test for divideIntoSegments
        '''
        
        cwd = os.path.dirname(os.path.realpath(__file__));
        tmp_dir = os.path.join(cwd, 'ut_divideIntoSegments');
        
        # Create new temp directory for each test
        try:
            os.stat(tmp_dir);
        except:
            os.mkdir(tmp_dir);
        shutil.rmtree(tmp_dir)
        os.mkdir(tmp_dir)
        
        exampleDir,imageDir = self.getExampleImage()
        for d in imageDir:
            t_dir = os.path.join(tmp_dir, d)
            os.mkdir(t_dir)
            
            d_dir = os.path.join(exampleDir, d)
            _,_, imageList = walk(d_dir).next()
            for f in imageList:
                img = cv2.imread(os.path.join(d_dir, f))
                if(len(img.shape)==3):
                    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
                _,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)            
                img = scaleToFill(img)
                _,image = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                segments = divideIntoSegments(5, image)
                i = 0
                fname = f.split('.')[0]+'_'
                for s in segments:
                    self.writeImageToDisk(os.path.join(t_dir, fname)+str(i)+".png", s)
                    i = i +1
    
    def test_extractSortedComponentSizeList(self):
        '''
        Unit-test for extractSortedComponentSizeList
        '''
        cwd = os.path.dirname(os.path.realpath(__file__));
        tmp_dir = os.path.join(cwd, 'ut_extractSortedComponentSizeList');
        
        # Create new temp directory for each test
        try:
            os.stat(tmp_dir)
        except:
            os.mkdir(tmp_dir)
        shutil.rmtree(tmp_dir)
        os.mkdir(tmp_dir)
        
        exampleDir,imageDir = self.getExampleImage()
        for d in imageDir:
            t_dir = os.path.join(tmp_dir, d)
            os.mkdir(t_dir)
            
            d_dir = os.path.join(exampleDir, d)
            _,_, imageList = walk(d_dir).next()
            for f in imageList:
                fname = f.split('.')[0]+'.txt'
                txtFile = open(os.path.join(t_dir, fname),'w+')
                
                img = cv2.imread(os.path.join(d_dir, f))
                if(len(img.shape)==3):
                    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                    
                _,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                img = scaleToFill(img)
                _,image = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                segments = divideIntoSegments(5, image)
                
                for s in segments:
                    component_size_list = extractSortedComponentSizeList(s)
                    txtFile.write(str(component_size_list))
                    txtFile.write('\n')
                
                txtFile.close()
    
    def test_sortedCCsSizeList(self):
        '''
        Unit-test for sortedCCsSizeList
        '''
        cwd = os.path.dirname(os.path.realpath(__file__));
        tmp_dir = os.path.join(cwd, 'ut_sortedCCsSizeList')
        
        # Create new temp directory for each test
        try:
            os.stat(tmp_dir)
        except:
            os.mkdir(tmp_dir)
        shutil.rmtree(tmp_dir)
        os.mkdir(tmp_dir)
        
        exampleDir,imageDir = self.getExampleImage()
        for d in imageDir:
            t_dir = os.path.join(tmp_dir, d)
            os.mkdir(t_dir)
            
            d_dir = os.path.join(exampleDir, d)
            _,_, imageList = walk(d_dir).next()
            for f in imageList:
                fname = f.split('.')[0]
                txtFile = open(os.path.join(t_dir, fname+'.txt'),'w+')
                
                img = cv2.imread(os.path.join(d_dir, f))
                if(len(img.shape)==3):
                    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                
                _,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                img = scaleToFill(img)
                _,image = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                segments = divideIntoSegments(5, image)
                
                i=0
                for s in segments:
                    #self.writeImageToDisk(os.path.join(t_dir, fname+'_')+str(i)+".png", s)
                    component_size_list = sortedCCsSizeList(s)
                    txtFile.write(str(component_size_list))
                    txtFile.write('\n')
                    i +=1
                
                txtFile.close()

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_word_']
    unittest.main() 