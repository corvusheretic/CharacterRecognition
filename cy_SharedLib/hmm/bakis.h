/*
 * baumwelch.h
 *
 *  Created on: Dec 18, 2015
 *      Author: kalyan
 */

#ifndef BAKIS_H_
#define BAKIS_H_


#include <vector>
#include <math.h>

#include "../common/globaldefs.h"

#define NUMPY_EPSILON 2.2204460492503131e-16

class Bakis {
public:

    /* Constructor */
	Bakis ( int *szDefs, int *cyO,
			HMMReal *cyA,	HMMReal *cyB,
			HMMReal *cyAlpha, HMMReal *cyBeta,
			HMMReal *cyNumA, HMMReal *cyDenA,
			HMMReal *cyNumB, HMMReal *cyDenB
			);
	~Bakis();

    /* Update probabilities as per Baum-Welch Algorithm*/
    void getBakis();

private:

    int N;
    int nCO, nRA, nCA, nRB, nCB;

    int  *ptrO;
    HMMReal *ptrA;
    HMMReal *ptrB;
    HMMReal *ptrAlpha;
    HMMReal *ptrBeta;

    HMMReal *ptrNumA;
    HMMReal *ptrDenA;
    HMMReal *ptrNumB;
    HMMReal *ptrDenB;

    HMMReal P;

	void updateA();
	void updateB();
};

#endif /* BAKIS_H_ */
