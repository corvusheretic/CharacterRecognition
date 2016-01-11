/*
 * calcforward.h
 *
 *  Created on: Dec 18, 2015
 *      Author: kalyan
 */

#ifndef CALCFORWARD_H_
#define CALCFORWARD_H_


#include <vector>
#include <math.h>

#include "../common/globaldefs.h"

class CalcForward {
public:

    /* Constructor */
	CalcForward ( int *szDefs, int *cyO,
			HMMReal *cyA, HMMReal *cyB, HMMReal *cyPI,
			HMMReal *cySF, HMMReal *cyAlpha );

	~CalcForward();

    /* Calculates the forward path probabilities */
    void getCalcForward();

private:

    int N;
    int nCO, nRA, nCA, nRB, nCB, nCPI;

    int  *ptrO;
    HMMReal *ptrA;
    HMMReal *ptrB;
    HMMReal *ptrPI;
    HMMReal *ptrSF;
    HMMReal *ptrAlpha;
};


#endif /* CALCFORWARD_H_ */
