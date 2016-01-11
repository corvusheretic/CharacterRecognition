'''
Created on Jan 2, 2016

@author: kalyan
'''

import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import shutil
from debug.debugger import db_StallExec

from os import walk
import numpy as np
import cv2

def getExampleImage():
    cwd         = os.path.dirname(os.path.realpath(__file__))
    #example_dir = os.path.join(cwd, '../../character_examples')
    #example_dir = os.path.join(cwd, '../../ut_SyntheticData')
    example_dir = os.path.join(cwd, "../../UniPenn")
    #example_dir = os.path.join(cwd, "../../debugImg")
    
    _,image_dir,_ = walk(example_dir).next()
    
    return (example_dir,image_dir)

def imgRegion(raster):
    
    indx = np.nonzero(raster < 1)
    
    min_x = np.min(indx[0])
    max_x = np.max(indx[0])
    min_y = np.min(indx[1])
    max_y = np.max(indx[1])
    
    return(raster[min_x:max_x+1,min_y:max_y+1])

def writeImageToDisk(image_path, image):
    cv2.imwrite(image_path,image)

def addImgBorder(src,
              margin):
    
    rows,cols = np.shape(src)
    dst = 255*np.ones((rows+2*margin,cols+2*margin)
                      , np.uint8)
    dst[margin:rows+margin, margin:cols+margin] = src
      
    return dst

def rotateImg(img,
              angle):
    
    src = imgRegion(img)
    rows,cols = np.shape(src)
    maxSide = max(cols,rows)
    src = addImgBorder(src, int(0.25*maxSide))
    
    kernel   = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    dilation = cv2.dilate(255-src,kernel,iterations = 4)
    
    #src = imgRegion(255-dilation)
    rows,cols = np.shape(dilation)
    
    if(1):
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        dst = cv2.warpAffine(dilation,M,(cols,rows))
    else:
        M = cv2.getRotationMatrix2D((rows/2,cols/2),angle,1)
        dst = cv2.warpAffine(dilation,M,(rows,cols))
    
    dst = imgRegion(255-dst)
    
    return dst

def skewImg(img,
              skewness,
              skewType):
    
    src = imgRegion(img)
    rows,cols = np.shape(src)
    maxSide = max(cols,rows)
    src = addImgBorder(src, 0.5*maxSide)
    
    kernel   = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    dilation = cv2.dilate(255-src,kernel,iterations = 2)
    
    #src = 255-dilation
    rows,cols = np.shape(dilation)
    
    if(skewType=='u'):
        origPts  = np.float32([[0,0],[0,cols-1],[rows-1,cols-1]])
        transPts = np.float32([[0,0],[0,cols-1],[rows-1,cols-1-int(skewness*cols/100.0)]])
    if(skewType=='d'):
        origPts  = np.float32([[0,0],[0,cols-1],[rows-1,0]])
        transPts = np.float32([[0,0],[0,cols-1],[rows-1,int(skewness*cols/100.0)]])
    if(skewType=='l'):
        origPts  = np.float32([[0,0],[0,cols-1],[rows-1,cols-1]])
        transPts = np.float32([[int(skewness*cols/100.0),0],[0,cols-1],[rows-1,cols-1]])
    if(skewType=='r'):
        origPts  = np.float32([[0,0],[rows-1,0],[0,cols-1]])
        transPts = np.float32([[0,0],[rows-1,0],[int(skewness*cols/100.0),cols-1]])
            
    M = cv2.getAffineTransform(origPts,transPts)
    dst = cv2.warpAffine(dilation,M,(cols,rows))
    
    dst = imgRegion(255-dst)
    
    return dst

def addNoise(img,
             noiseLevel):
    src = imgRegion(img)
    src = addImgBorder(src, MARGINSIZE)
    
    noise = np.zeros(src.shape)
    cv2.randu(noise,0,255)
    noise = (255.0*(noise>noiseLevel)).astype(np.uint8)
    
    kernel   = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    dilation = cv2.dilate(255-src,kernel,iterations = 1)
    
    noise    = noise & dilation
    
    dst = src & (255-noise)
    dst = imgRegion(dst)
    
    return dst

def resizeWithAspectRatio(img,
                          resize):
    if(len(img.shape)==3):
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
    _,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    src = imgRegion(img)
    src = addImgBorder(src, MARGINSIZE)
    
    kernel   = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    dilation = cv2.dilate(255-src,kernel,iterations = 2)
    
    src = imgRegion(255-dilation)
    rows,cols = np.shape(src)
    maxSide = max(cols,rows)
    
    dst = (255.0*np.ones((resize,resize))).astype(np.uint8)
    
    if(maxSide == rows):
        altSz = int(np.floor(float(resize)/rows*cols))
        rImg = cv2.resize(img,(resize,altSz))
        _,rImg = cv2.threshold(rImg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        margin = int(np.floor((resize - altSz)/2.0))
        dst[margin:altSz+margin,:] = rImg
    else:
        altSz = int(np.floor(float(resize)/cols*rows))
        rImg = cv2.resize(img,(altSz,resize))
        _,rImg = cv2.threshold(rImg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        margin = int(np.floor((resize - altSz)/2.0))
        dst[:,margin:altSz+margin] = rImg
    
    return(dst)

ROTATION    = 15
#SK_ROTATION = 10
SKEWNESS    = 20.0
NOISELEVEL  = 64 #std. deviation of white noise
MORPH_SIZE  = 3 #size of Structuring Element
MARGINSIZE  = 500 # pixels border to avoid truncation
STDSIZE     = 100

if __name__ == "__main__":
    
    cwd      = os.path.dirname(os.path.realpath(__file__));
    base_dir,imageDirs = getExampleImage()
    test_dir = os.path.join(cwd, '../../AdvSyntheticData')
    
    skewDir  = ['n','l','r','u','d']
    rotDir   = [-1,0,1]
    noiseDir = [0,1]

    
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
            srcName,extn = f.split('.')
            img = cv2.imread(os.path.join(d_dir, f))
            if(len(img.shape)==3):
                img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            
            _,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            
            for ang in rotDir:
                db_StallExec(0)
                rimg = rotateImg(img, ROTATION*ang)
                if(ang == -1):
                    dstNameR = srcName + '_' + 'c' + str(ROTATION)
                if(ang == 1):
                    dstNameR = srcName + '_' + 'a' + str(ROTATION)
                if(ang == 0):
                    dstNameR = srcName
                
                for sk in skewDir:
                    if(sk == 'n'):
                        skImg = rimg
                        dstNameS = dstNameR
                    else:
                        skImg = skewImg(rimg, SKEWNESS, sk)
                        dstNameS = dstNameR + '_' + 'sk' + sk.upper()
                    
                    for no in noiseDir:
                        if(no):
                            nImg = addNoise(skImg,NOISELEVEL)
                            dstNameN = dstNameS + '_' + 'noisy'
                        else:
                            nImg = skImg
                            dstNameN = dstNameS
                    
                        dst = resizeWithAspectRatio(nImg, STDSIZE)
                        dstName = dstNameN + '.' + extn
                        writeImageToDisk(os.path.join(t_dir, dstName), dst)
                        
                        db_StallExec(0)
