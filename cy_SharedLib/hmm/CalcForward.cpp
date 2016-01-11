/*
 * CalcForward.cpp
 *
 *  Created on: Dec 18, 2015
 *      Author: kalyan
 */
#include <iostream>
#include <algorithm>    // std::generate

#include "calcforward.h"

/* Constructor */
CalcForward::CalcForward ( int *szDefs, int *cyO,
		HMMReal *cyA, HMMReal *cyB, HMMReal *cyPI,
		HMMReal *cySF, HMMReal *cyAlpha )
{
	N    = szDefs[0];
	nCO  = szDefs[1];
	nRA  = szDefs[2];
	nCA  = szDefs[3];
	nRB  = szDefs[4];
	nCB  = szDefs[5];
	nCPI = szDefs[6];

	ptrO = cyO;
	ptrA = cyA;
	ptrB = cyB;
	ptrPI = cyPI;
	ptrSF = cySF;
	ptrAlpha = cyAlpha;
}

CalcForward::~CalcForward ()
{

}

void CalcForward::getCalcForward()
{

	HMMReal scaling_factor;
	HMMReal sum_alpha;

	//initialization
	lmat matO(ptrO, nCO, 1, false);
	rmat matA(ptrA, nCA, nRA, false);
	rmat matB(ptrB, nCB, nRB, false);
	rmat matPI(ptrPI, nCPI, 1, false);
	rmat matSF(ptrSF, nCO, 1, false);
	rmat matAlpha(ptrAlpha, N, nCO, false);


	unsigned int t = 0;
	int otCol = matO(t);

	matAlpha.col(t) = matPI.col(0) % arma::trans(matB.row(otCol));

	//induction
	for (t=1; t < matO.n_elem; t++)
	{
		otCol = matO(t);

		sum_alpha = arma::sum(matAlpha.col(t-1));
		scaling_factor = (sum_alpha) ? 1.0/sum_alpha : 1.0;
		matAlpha.col(t-1) = scaling_factor * matAlpha.col(t-1);

		rmat prob_sum = matA * matAlpha.col(t-1);
		matAlpha.col(t) = prob_sum % arma::trans(matB.row(otCol));
		matSF(t-1) = scaling_factor;
	}

	sum_alpha = arma::sum(matAlpha.col(t-1));
	scaling_factor = (sum_alpha) ? 1.0/sum_alpha : 1.0;
	matAlpha.col(t-1) = scaling_factor * matAlpha.col(t-1);
	matSF(t-1) = scaling_factor;

#if __CY_DEBUG_PRINT__
	matAlpha.print("matAlpha: ");
#endif
}
