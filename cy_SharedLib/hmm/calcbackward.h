/*
 * calcbackward.h
 *
 *  Created on: Dec 18, 2015
 *      Author: kalyan
 */

#ifndef CALCBACKWARD_H_
#define CALCBACKWARD_H_


#include <vector>
#include <math.h>

#include "../common/globaldefs.h"

class CalcBackward {
public:

    /* Constructor */
	CalcBackward ( int *szDefs, int *cyO,
			HMMReal *cyA,	HMMReal *cyB,
			HMMReal *cySF, HMMReal *cyBeta );
	~CalcBackward();

    /* Calculates the forward path probabilities */
    void getCalcBackward();

private:

    int N;
    int nCO, nRA, nCA, nRB, nCB;

    int   *ptrO;
    HMMReal *ptrA;
    HMMReal *ptrB;
    HMMReal *ptrSF;
    HMMReal *ptrBeta;
};


#endif /* CALCBACKWARD_H_ */
