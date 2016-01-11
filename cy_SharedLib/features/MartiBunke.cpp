/*
 * cyMartiBunke.cpp
 *
 *  Created on: Dec 11, 2015
 *      Author: kalyan
 */
#include <iostream>
#include <algorithm>    // std::generate
#include "martibunke.h"

struct c_unique {
	int current;
	c_unique() {current=0;}
	int operator()() {return current++;}
} UniqueNumber;

/* Constructor */
MartiBunke::MartiBunke ( float *cyData,
						int Rows, int Cols,
						float *mbFeature, double th )
{
    nCols = Cols;
    nRows = Rows;
    ptrData = cyData;
    thr = (float)th;
    //cv::threshold(img, binImg, thr, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    mbFeatMat = mbFeature;//arma::trans(arma::fmat(mbFeature, nCols, 9, true));
}

MartiBunke::~MartiBunke()
{

}

/* Method to calculate the Marti & Bunke features */
void MartiBunke::getMartiBunke() {

    /* Feature vector */
#if __ROW_MAJOR_ACCESS__
	arma::fmat img = arma::trans(arma::fmat(ptrData, nCols, nRows, true));
	std::vector<float> indxVec(img.n_rows);
#else
	arma::fmat img(ptrData, nCols, nRows, false);
	std::vector<float> indxVec(img.n_cols);
#endif


#if __CY_DEBUG_PRINT__
    img.print("Img: ");
#endif

    generate(indxVec.begin(), indxVec.end(), UniqueNumber);

    /* Loops through each column of the window since features
     * are extracted using a sliding window */
#if __ROW_MAJOR_ACCESS__
    for ( unsigned int j = 0; j < img.n_cols; j++ ) {

        /* Create a copy of the current column */
    	jCol = img.col(j);
#else
    for ( unsigned int j = 0; j < img.n_rows; j++ ) {
    	jCol = arma::trans(img.row(j));
#endif

    	std::vector<float> f;

#if __CY_DEBUG_PRINT__
    	jCol.print("Col: ");
#endif

    	/* Get all of the features */
        getF1F2F3(indxVec,f);
        getF4F5F6F7(f);
        getF8(f);
		getF9(f);
#if __CY_DEBUG_PRINT__
        std::cout << "MB-Feature for column: " << j << " is: ";
        for (std::vector<float>::const_iterator i = f.begin(); i != f.end(); ++i)
            std::cout << *i << ' '; // use iterator if values of f change
        std::cout << std::endl;
#endif
        std::memcpy(mbFeatMat + j*f.size(),
        		f.data(), f.size()*sizeof(float));
    }

}

/* Calculates features F1,F2,F3.
 * F1 feature is the weight of the window
 * F2 feature is the center of gravity of the window
 * F3 feature is the second order moment of the window
 */
void MartiBunke::getF1F2F3(std::vector<float> indxVec,
		std::vector<float> &mbFeatVec) {
	arma::fmat indxMat = arma::fmat(indxVec.data(),indxVec.size(), 1, true);
	arma::fmat outMat  = arma::fmat(indxVec.size(), 1);

    /* Get the mean pixel value in column window */
    float f1_t = arma::mean(arma::vectorise( jCol ));

    /* Compute the first order moment of the column vector */
    outMat = jCol % indxMat;

	float f2_t = arma::mean(arma::vectorise( outMat ));

    /* Compute the second order moment of the column vector */
    outMat = outMat % indxMat;

    float f3_t = arma::mean(arma::vectorise( outMat ));

    /* Return the feature */
    mbFeatVec.push_back(f1_t);
    mbFeatVec.push_back(f2_t);
    mbFeatVec.push_back(f3_t/(1.0 * indxVec.size()));
}


/* Calculates the F4 and F5 features.
 * F4 feature is the upper position of the contour of the window
 * F5 feature is the lower position of the contour of the window
 * F6 feature is the gradient of the upper contour of the window
 * F7 feature is the gradient of the lower contour of the window
 */
void MartiBunke::getF4F5F6F7(std::vector<float> &mbFeatVec) {

#if __ROW_MAJOR_ACCESS__
    double f4_t = nRows-1; // Assume the upper contour is at the bottom
#else
	double f4_t = nCols-1; // Assume the upper contour is at the bottom
#endif
    double f5_t = 0;            // Assume the lower contour is at the top
    /* If a foreground pixel is found, change the upper contour */
    arma::uvec idx = arma::find(jCol <= thr); // Assuming a sorted array

#if __CY_DEBUG_PRINT__
    jCol.print("col:");
    std::cout << "Thr:" << thr << std::endl;
    idx.print("Idx:");
    std::cout << "No.elem:" << idx.n_elem << std::endl;
#endif

    if(idx.n_elem > 0)
    {
    	f4_t = idx(0); // Upper Contour
    	f5_t = idx(idx.size()-1); // Lower Contour
    }

    //Assume adjacent pixel to border as BG on the top as well as bottom
    float f6_t = (f4_t > 0) ? (jCol[f4_t-1] - jCol[f4_t])
    		: (1.0 - jCol[f4_t]); // Gradient of the upper contour
#if __ROW_MAJOR_ACCESS__
    float f7_t = (f5_t < (nRows-1)) ? (jCol[f5_t+1] - jCol[f5_t])
    		: (1.0 - jCol[f5_t]); // Gradient of the lower contour
#else
    float f7_t = (f5_t < (nCols-1)) ? (jCol[f5_t+1] - jCol[f5_t])
        		: (1.0 - jCol[f5_t]); // Gradient of the lower contour
#endif

    mbFeatVec.push_back(f4_t);
    mbFeatVec.push_back(f5_t);
    mbFeatVec.push_back(f6_t);
    mbFeatVec.push_back(f7_t);
}

/* Calculates the F8 feature.
 * F8 feature is the number of background-foreground transitions
 */
void MartiBunke::getF8(std::vector<float> &mbFeatVec) {

    arma::fmat binCol  = jCol;
    binCol.fill(thr);
    arma::umat binCol1 = jCol > binCol;
    arma::umat numZ = binCol1;

    numZ.rows(0,jCol.n_rows-2) = binCol1.rows(1,jCol.n_rows-1);
    numZ(jCol.n_rows-1,0) = binCol1(jCol.n_rows-1,0);
    //numZ.print("numZ:");
    //binCol1.print("binCol1:");

    arma::uvec vecTrans = arma::nonzeros(numZ - binCol1);
    float f8_t = vecTrans.n_elem;

    mbFeatVec.push_back(f8_t);
}

/* Calculates the F9 feature.
 * F9 feature is the number of foreground pixels in between the
 * upper and lower contour divided by the height of the contour.
 */
void MartiBunke::getF9(std::vector<float> &mbFeatVec) {

    arma::fmat binCol  = jCol;
    binCol.fill(thr);
    arma::umat binCol1 = jCol < binCol;
    arma::umat numZ;

    int uContour = mbFeatVec.at(3);
    int lContour = mbFeatVec.at(4);

    numZ = (uContour < lContour) ? binCol1.rows(uContour,lContour) :
    		binCol1.rows(lContour,uContour); //Case when no FG pixels detected

    arma::uvec vecTrans = arma::nonzeros(numZ);
    float numElems = vecTrans.n_elem ? (vecTrans.n_elem - 1) : vecTrans.n_elem ;
    float f9_t        = numElems /(std::abs(uContour-lContour) + 1.0);

    mbFeatVec.push_back(f9_t);
}

