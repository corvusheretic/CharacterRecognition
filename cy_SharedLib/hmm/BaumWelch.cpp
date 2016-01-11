/*
 * BaumWelch.cpp
 *
 *  Created on: Dec 18, 2015
 *      Author: kalyan
 */
#include <iostream>
#include <algorithm>    // std::generate

#include "baumwelch.h"

/* Constructor */
BaumWelch::BaumWelch ( int *szDefs, int *cyO,
		HMMReal *cyA,	HMMReal *cyB,
		HMMReal *cyPI,	HMMReal *cySF,
		HMMReal *cyAlpha, HMMReal *cyBeta )
{
	N    = szDefs[0];
	nCO  = szDefs[1];
	nRA  = szDefs[2];
	nCA  = szDefs[3];
	nRB  = szDefs[4];
	nCB  = szDefs[5];

	ptrO     = cyO;
	ptrA     = cyA;
	ptrB     = cyB;
	ptrPI    = cyPI;
	ptrSF    = cySF;
	ptrAlpha = cyAlpha;
	ptrBeta  = cyBeta;
}

BaumWelch::~BaumWelch()
{

}

void BaumWelch::getBaumWelch()
{
	//initialization
/*
	xi = np.zeros((len(O) - 1, N, N))
	gamma = np.zeros((len(O) - 1, N))
*/
	xi    = rcube(N,N,(nCO-1));
#if __ROW_MAJOR_ACCESS__
	gamma = rmat((nCO-1), N);
#else
	gamma = rmat(N,(nCO-1));
#endif

	getXiGamma();
	updatePiAB();
}

void BaumWelch::getXiGamma()
{

#if __ROW_MAJOR_ACCESS__
	lmat matO     = arma::trans(lmat(ptrO, nCO, 1, false));
	rmat matA     = arma::trans(rmat(ptrA, nCA, nRA, false));
	rmat matB     = arma::trans(rmat(ptrB, nCB, nRB, false));
	rmat matAlpha = arma::trans(rmat(ptrAlpha, N, nCO, false));
	rmat matBeta  = arma::trans(rmat(ptrBeta, N, nCO, false));
#else
	lmat matO(ptrO, nCO, 1, false);
	rmat matA(ptrA, nCA, nRA, false);
	rmat matB(ptrB, nCB, nRB, false);
	rmat matAlpha(ptrAlpha, N, nCO, false);
	rmat matBeta(ptrBeta, N, nCO, false);
#endif

	int t=0,i;
	int otCol;
	HMMReal sum;

/*
    for t in range(len(O) - 1):
        s = 0
        for i in range(N):
            for j in range(N):
                xi[t][i][j] = alpha[t][i] * A[i][j] * B[j][O[t+1]] * beta[t+1][j]
                s += xi[t][i][j]
        # Normalize
        for i in range(N):
            for j in range(N):
                xi[t][i][j] *= 1/s
 */
	rmat matBCol     = rmat(N,N);
	rmat matAlphaRow = rmat(N,N);
	rmat matBetaRow  = rmat(N,N);

#if __ROW_MAJOR_ACCESS__
	for (t = 0; t < nCO-1 ; t++)
	{
		otCol = matO(t+1);
		rmat BCol = arma::trans(matB.col(otCol));
		matBCol.row(0) = BCol;

		rmat alphaRow = arma::trans(matAlpha.row(t)); // make it a row for vstack
		rmat betaRow  = matBeta.row(t+1);

		matAlphaRow.col(0) = alphaRow;
		matBetaRow.row(0)  = betaRow;

		for (i = 1; i < N ; i++)
		{
			matBCol.row(i)     = BCol;
			matAlphaRow.col(i) = alphaRow;
			matBetaRow.row(i)  = betaRow;
		}

		rmat slice = matAlphaRow % matA % matBCol % matBetaRow;
		sum = arma::accu(slice);

		xi.slice(t) = slice/sum;

		/*
		 # Now calculate the gamma table
		for t in range(len(O) - 1):
			for i in range(N):
				s = 0
				for j in range(N):
					s += xi[t][i][j]
				gamma[t][i] = s

		 */
		gamma.row(t) = arma::trans(arma::sum(slice,1))/sum;
	}
#else

	for (t = 0; t < nCO-1 ; t++)
	{
		otCol = matO(t+1);
		rmat BCol = arma::trans(matB.row(otCol));
		matBCol.col(0) = BCol;

		rmat alphaRow = arma::trans(matAlpha.col(t)); // make it a row for vstack
		rmat betaRow  = matBeta.col(t+1);

		matAlphaRow.row(0) = alphaRow;
		matBetaRow.col(0)  = betaRow;

		for (i = 1; i < N ; i++)
		{
			matBCol.col(i)     = BCol;
			matAlphaRow.row(i) = alphaRow;
			matBetaRow.col(i)  = betaRow;
		}

		rmat slice = matAlphaRow % matA % matBCol % matBetaRow;
		sum = arma::accu(slice);

		xi.slice(t) = slice/sum;

		/*
		 # Now calculate the gamma table
		for t in range(len(O) - 1):
			for i in range(N):
				s = 0
				for j in range(N):
					s += xi[t][i][j]
				gamma[t][i] = s

		 */
		gamma.col(t) = arma::trans(arma::sum(slice,0))/sum;
	}

#endif

#if __CY_DEBUG_PRINT__
	xi.print("xi: ");
	gamma.print("gamma: ");
#endif
}

void BaumWelch::updatePiAB()
{

#if __ROW_MAJOR_ACCESS__
	lmat matO     = arma::trans(lmat(ptrO, nCO, 1, false));
	rmat matA     = arma::trans(rmat(ptrA, nCA, nRA, false));
	rmat matB     = arma::trans(rmat(ptrB, nCB, nRB, false));
	rmat matPi    = arma::trans(rmat(ptrPI, N, 1, false));
#else
	lmat matO(ptrO, nCO, 1, false);
	rmat matA(ptrA, nCA, nRA, false);
	rmat matB(ptrB, nCB, nRB, false);
	rmat matPi(ptrPI, N, 1, false);
#endif

	int t,i,j;

/*
    # Update pi
    for i in range(N):
        pi[i] = gamma[0][i]
 */
#if __ROW_MAJOR_ACCESS__
	matPi = gamma.row(0);
#else
	matPi = gamma.col(0);
#endif

#if __CY_DEBUG_PRINT__
	matPi.print("matPi: ");
#endif

/*
    # Update A
    #print 'Updating A'
    for i in range(N):
        for j in range(N):
            numerator = 0
            denominator = 0
            for t in range(len(O) - 1):
                numerator += xi[t][i][j]
                denominator += gamma[t][i]
            A[i][j] = numerator / denominator
 */

	rmat matGammaCSums = rmat(N,N);
	rmat matSliceSums  = rmat(N,N);
	matSliceSums.fill(0.0);

#if __ROW_MAJOR_ACCESS__
	rmat ColSum = arma::trans(arma::sum(gamma,0));
	for (i = 0; i < N ; i++)
		matGammaCSums.col(i) = ColSum;

	for (t = 0; t < nCO-1 ; t++)
		matSliceSums += xi.slice(t);

	matA = matSliceSums / matGammaCSums;
#else
	rmat ColSum = arma::trans(arma::sum(gamma,1));
	for (i = 0; i < N ; i++)
		matGammaCSums.row(i) = ColSum;

	for (t = 0; t < nCO-1 ; t++)
		matSliceSums += xi.slice(t);

	matA = matSliceSums / matGammaCSums;
#endif

#if __CY_DEBUG_PRINT__
	matA.print("matA: ");
#endif

/*
     # Update B
    for j in range(N):
        for k in range(K):
            numerator = 0
            denominator = 0
            for t in range(len(O) - 1):
                if O[t] == k:
                    numerator += gamma[t][j]
                denominator += gamma[t][j]
            B[j][k] = numerator / denominator
 */
#if __ROW_MAJOR_ACCESS__
	rmat symMat   = 0.0 * gamma;
	rmat sumRow   = arma::sum(gamma,0);

	for (i = 0; i < nCB ; i++)
	{
		lmat redO = matO.cols(0,matO.n_elem-2);

		arma::umat symCol = arma::trans(redO == i);
		for(j=0; j< gamma.n_cols; j++)
			symMat.col(j) = arma::conv_to<rmat>::from(symCol);

		symMat = symMat % gamma;
		rmat symProb = arma::sum(symMat, 0);
		symProb = symProb / sumRow;

		matB.col(i) = arma::trans(symProb);
	}
#else
	rmat symMat   = 0.0 * gamma;
	rmat sumRow   = arma::sum(gamma,1);

	for (i = 0; i < nCB ; i++)
	{
		lmat redO = matO.rows(0,matO.n_elem-2);

		arma::umat symCol = arma::trans(redO == i);
		for(j=0; j< gamma.n_rows; j++)
			symMat.row(j) = arma::conv_to<rmat>::from(symCol);

		symMat = symMat % gamma;
		rmat symProb = arma::sum(symMat, 1);
		symProb = symProb / sumRow;

		matB.row(i) = arma::trans(symProb);
	}
#endif
#if __CY_DEBUG_PRINT__
	matB.print("matB: ");
#endif
}
