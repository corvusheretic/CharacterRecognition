/*
 * martibunke.h
 *
 *  Created on: Dec 12, 2015
 *      Author: kalyan
 */

/*

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

*/

/*Features based on:
 *
 * U.-V. Marti. Off-line recognition of handwritten texts. PhD thesis, University of Bern, Switzerland, 2000.
 *
 * U.-V. Marti and H. Bunke. Using a statistical language model to improve the performance of an HMM-based
 * cursive handwriting recognition systems. In Hidden Markov models: applications in computer vision,
 * pages 65–90. World Scientiﬁc Publishing Co., Inc., 2002.
 */

/* This class is used to calculate the Marti & Bunke Feature set */

#ifndef MARTIBUNKE_H_
#define MARTIBUNKE_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include <math.h>

#include "../common/globaldefs.h"

class MartiBunke {
public:

    /* Constructor */
	MartiBunke ( float *cyData, int nRows, int nCols,
			float *mbFeature, double thr );

	~MartiBunke();

    /* Calculates the features */
    void getMartiBunke();

private:

    int nCols, nRows;
    float *ptrData;
    float *mbFeatMat;
    arma::fmat jCol;
    float thr;

    /* Private methods for calculating features */
    void getF1F2F3(std::vector<float> indxVec, std::vector<float> &mbFeatVec);
    void getF4F5F6F7(std::vector<float> &mbFeatVec);
    void getF8(std::vector<float> &mbFeatVec);
    void getF9(std::vector<float> &mbFeatVec);

};

#endif /* MARTIBUNKE_H_ */
