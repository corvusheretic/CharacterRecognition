/*
 * baumwelch.h
 *
 *  Created on: Dec 18, 2015
 *      Author: kalyan
 */

#ifndef BAUMWELCH_H_
#define BAUMWELCH_H_


#include <vector>
#include <math.h>

#include "../common/globaldefs.h"

class BaumWelch {
public:

    /* Constructor */
	BaumWelch ( int *szDefs, int *cyO,
			HMMReal *cyA,	HMMReal *cyB,
			HMMReal *cyPI,	HMMReal *cySF,
			HMMReal *cyAlpha, HMMReal *cyBeta );
	~BaumWelch();

    /* Update probabilities as per Baum-Welch Algorithm*/
    void getBaumWelch();

private:

    int N;
    int nCO, nRA, nCA, nRB, nCB;

    int  *ptrO;
    HMMReal *ptrA;
    HMMReal *ptrB;
    HMMReal *ptrPI;
    HMMReal *ptrSF;
    HMMReal *ptrAlpha;
    HMMReal *ptrBeta;

    rcube xi;
    rmat gamma;

    void getXiGamma();
    void updatePiAB();
};

#endif /* BAUMWELCH_H_ */
