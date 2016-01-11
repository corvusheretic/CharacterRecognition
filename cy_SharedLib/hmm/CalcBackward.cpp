/*
 * CalcBackward.cpp
 *
 *  Created on: Dec 18, 2015
 *      Author: kalyan
 */
#include <iostream>
#include <algorithm>    // std::generate

#include "calcbackward.h"

/* Constructor */
CalcBackward::CalcBackward ( int *szDefs, int *cyO,
		HMMReal *cyA,	HMMReal *cyB,
		HMMReal *cySF, HMMReal *cyBeta )
{
	N    = szDefs[0];
	nCO  = szDefs[1];
	nRA  = szDefs[2];
	nCA  = szDefs[3];
	nRB  = szDefs[4];
	nCB  = szDefs[5];

	ptrO = cyO;
	ptrA = cyA;
	ptrB = cyB;
	ptrSF = cySF;
	ptrBeta = cyBeta;
}

CalcBackward::~CalcBackward()
{

}

void CalcBackward::getCalcBackward()
{

/*
    T = len(O)
    beta = np.zeros((T, N))
    #initialization
    for i in range(N):
        beta[T-1][i] = 1.0
 */
	HMMReal scaling_factor;

	//initialization
#if __ROW_MAJOR_ACCESS__
	lmat matO          = arma::trans(lmat(ptrO, nCO, 1, false));
	rmat matA    = arma::trans(rmat(ptrA, nCA, nRA, false));
	rmat matB    = arma::trans(rmat(ptrB, nCB, nRB, false));
	rmat matSF   = arma::trans(rmat(ptrSF, nCO, 1, false));
	rmat matBeta = arma::trans(rmat(ptrBeta, N, nCO, false));
	matBeta.fill(1.0);
#else
	lmat matO(ptrO, nCO, 1, false);
	rmat matA(ptrA, nCA, nRA, false);
	rmat matB(ptrB, nCB, nRB, false);
	rmat matSF(ptrSF, nCO, 1, false);
	rmat matBeta(ptrBeta, N, nCO, false);
	matBeta.fill(1.0);
#endif

	int t;
	int otCol;

/*
     for t in range(T-2, -1, -1):
        for i in range(self.N):
            prob_sum = 0
            for j in range(self.N):
                prob_sum += A[i][j] * (scaling_factor[t+1] *B[j][O[t+1]]) * beta[t+1][j]
            beta[t][i] = prob_sum

 */
#if __ROW_MAJOR_ACCESS__
	for (t = nCO-2; t > -1 ; t--)
	{
		scaling_factor = matSF(t+1);
		otCol = matO(t+1);
		rmat prob_sum = 	scaling_factor *
								(arma::trans(matBeta.row(t+1)) % matB.col(otCol));
		prob_sum = matA * prob_sum;
		matBeta.row(t) = arma::trans(prob_sum);
	}
#else
	for (t = nCO-2; t > -1 ; t--)
	{
		scaling_factor = matSF(t+1);
		otCol = matO(t+1);
		rmat prob_sum = 	scaling_factor *
								(arma::trans(matBeta.col(t+1)) % matB.row(otCol));
		prob_sum = prob_sum * matA;
		matBeta.col(t) = arma::trans(prob_sum);
	}

#endif

#if __CY_DEBUG_PRINT__
	matBeta.print("matBeta: ");
#endif
}
