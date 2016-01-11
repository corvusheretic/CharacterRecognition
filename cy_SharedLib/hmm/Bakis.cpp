/*
 * BaumWelch.cpp
 *
 *  Created on: Dec 18, 2015
 *      Author: kalyan
 */
#include <iostream>
#include <algorithm>    // std::generate

#include "bakis.h"

/* Constructor */
Bakis::Bakis ( int *szDefs, int *cyO,
		HMMReal *cyA,	HMMReal *cyB,
		HMMReal *cyAlpha, HMMReal *cyBeta,
		HMMReal *cyNumA, HMMReal *cyDenA,
		HMMReal *cyNumB, HMMReal *cyDenB
		)
{
	N       = szDefs[0];
	nCO     = szDefs[1];
	nRA     = szDefs[2];
	nCA     = szDefs[3];
	nRB     = szDefs[4];
	nCB     = szDefs[5];

	ptrO     = cyO;
	ptrA     = cyA;
	ptrB     = cyB;
	ptrAlpha = cyAlpha;
	ptrBeta  = cyBeta;

	ptrNumA  = cyNumA;
	ptrDenA  = cyDenA;
	ptrNumB  = cyNumB;
	ptrDenB  = cyDenB;

	P        = 0.0;
}

Bakis::~Bakis()
{

}

void Bakis::getBakis()
{
	updateA();
	updateB();
}

void Bakis::updateA()
{
#if __ROW_MAJOR_ACCESS__
	lmat matO     = arma::trans(lmat(ptrO, nCO, 1, false));
	rmat matA     = arma::trans(rmat(ptrA, nCA, nRA, false));
	rmat matB     = arma::trans(rmat(ptrB, nCB, nRB, false));
	rmat matAlpha = arma::trans(rmat(ptrAlpha, N, nCO, false));
	rmat matBeta  = arma::trans(rmat(ptrBeta, N, nCO, false));

	rmat matNumA  = arma::trans(rmat(ptrNumA, nCA, nRA, false));
	rmat matDenA  = arma::trans(rmat(ptrDenA, nCA, nRA, false));
#else
	lmat matO(ptrO, nCO, 1, false);
	rmat matA(ptrA, nCA, nRA, false);
	rmat matB(ptrB, nCB, nRB, false);
	rmat matAlpha(ptrAlpha, N, nCO, false);
	rmat matBeta(ptrBeta, N, nCO, false);

	rmat matNumA(ptrNumA, nCA, nRA, false);
	rmat matDenA(ptrDenA, nCA, nRA, false);
#endif

	int t=0,i;
	int otCol;

/*
	P = 0
	T = len(O[k])

	for i in range(N):
		P += alpha[T-1][i]

	if( P ==0 ): # Hack to avoid division by zero later
		P = np.finfo(float).eps
 */

	//matAlpha.print("matAlpha: ");

#if __ROW_MAJOR_ACCESS__
	P = arma::sum(matAlpha.row(matAlpha.n_rows-1));
#else
	P = arma::sum(matAlpha.col(matAlpha.n_cols-1));
#endif
	P = (P) ? P : NUMPY_EPSILON;

/*
 	nS = np.zeros((N,N),dtype=np.float64)
	dS = np.zeros((N,N),dtype=np.float64)
	for t in range(T - 1):
		for i in range(N):
			for j in range(N):
				nS[i][j] += alpha[t][i] * B[j][O[k][t+1]] * beta[t+1][j]
				dS[i][j] += alpha[t][i] * beta[t][i]
	sumNum_A += 1.0/P * nS
	sumDen_A += 1.0/P * dS
 */

	rmat matBCol     = rmat(N,N);
	rmat matAlphaRow = rmat(N,N);
	rmat matBetaRow  = rmat(N,N);

	rmat nS 		 = rmat(N,N);
	rmat dS 		 = rmat(N,N);
	nS.fill(0.0);
	dS.fill(0.0);

	rmat prodMat = matAlpha % matBeta;

#if __ROW_MAJOR_ACCESS__

	rmat gamma   = prodMat.submat(0,0,prodMat.n_rows-2,prodMat.n_cols-1);
	rmat sumRow  = arma::sum(gamma,0);

	for (t = 0; t < nCO-1 ; t++)
	{
		otCol = matO(t+1);
		rmat BCol = arma::trans(matB.col(otCol));
		matBCol.row(0) = BCol;

		rmat alphaRow = arma::trans(matAlpha.row(t)); // make it a row for vstack
		rmat betaRow  = matBeta.row(t+1);

		matAlphaRow.col(0) = alphaRow;
		matBetaRow.row(0)  = betaRow;

		dS.col(0) = arma::trans(sumRow);

		for (i = 1; i < N ; i++)
		{
			matBCol.row(i)     = BCol;
			matAlphaRow.col(i) = alphaRow;
			matBetaRow.row(i)  = betaRow;

			dS.col(i) = arma::trans(sumRow);
		}

		rmat nSlice = matAlphaRow % matBCol % matBetaRow;

		nS += nSlice;
	}

	matNumA += 1.0/P * nS;
	matDenA += 1.0/P * dS;

#else

	rmat gamma    = prodMat.submat(0,0,prodMat.n_rows-1,prodMat.n_cols-2);
	rmat sumRow   = arma::sum(gamma,1);

	for (t = 0; t < nCO-1 ; t++)
	{
		otCol = matO(t+1);
		rmat BCol = arma::trans(matB.row(otCol));
		matBCol.col(0) = BCol;

		rmat alphaRow = arma::trans(matAlpha.col(t)); // make it a row for vstack
		rmat betaRow  = matBeta.col(t+1);

		matAlphaRow.row(0) = alphaRow;
		matBetaRow.col(0)  = betaRow;

		dS.row(0) = arma::trans(sumRow);

		for (i = 1; i < N ; i++)
		{
			matBCol.col(i)     = BCol;
			matAlphaRow.row(i) = alphaRow;
			matBetaRow.col(i)  = betaRow;

			dS.row(i) = arma::trans(sumRow);
		}

		rmat nSlice = matAlphaRow % matBCol % matBetaRow;

		nS += nSlice;
	}

	matNumA += 1.0/P * nS;
	matDenA += 1.0/P * dS;

#endif

#if __CY_DEBUG_PRINT__
	matNumA.print("matNumA: ");
	matDenA.print("matDenA: ");
#endif

}

void Bakis::updateB()
{
#if __ROW_MAJOR_ACCESS__
	lmat matO     = arma::trans(lmat(ptrO, nCO, 1, false));
	rmat matAlpha = arma::trans(rmat(ptrAlpha, N, nCO, false));
	rmat matBeta  = arma::trans(rmat(ptrBeta, N, nCO, false));

	rmat matNumB  = arma::trans(rmat(ptrNumB, nCB, nRB, false));
	rmat matDenB  = arma::trans(rmat(ptrDenB, nCB, nRB, false));
#else
	lmat matO(ptrO, nCO, 1, false);
	rmat matAlpha(ptrAlpha, N, nCO, false);
	rmat matBeta(ptrBeta, N, nCO, false);

	rmat matNumB(ptrNumB, nCB, nRB, false);
	rmat matDenB(ptrDenB, nCB, nRB, false);
#endif

	int t=0,i;
	int otCol;

/*
	# Update B
	# Calculate the numerator and denominator
	#T = len(O[k])
	nS = np.zeros((N,K),dtype=np.float64)
	dS = np.zeros((N,K),dtype=np.float64)
	for j in range(N):
		for l in range(K):
			for t in range(T - 1):
				if O[k][t] == l:
					nS[j][l] += alpha[t][j] * beta[t][j]
				dS[j][l] += alpha[t][j] * beta[t][j]
	sumNum_B += 1.0 / P * nS
	sumDen_B += 1.0 / P * dS
*/

	rmat nS 		 = 0.0*matNumB;
	rmat dS 		 = 0.0*matDenB;

	rmat prodMat = matAlpha % matBeta;

#if __ROW_MAJOR_ACCESS__
	rmat gamma   = prodMat.submat(0,0,prodMat.n_rows-2,prodMat.n_cols-1);
	rmat symMat   = 0.0 * gamma;
	rmat sumRow   = arma::sum(gamma,0);

	for (i = 0; i < nCB ; i++)
	{
		lmat redO = matO.cols(0,matO.n_elem-2);

		arma::umat symCol = arma::trans(redO == i);
		for(t=0; t< gamma.n_cols; t++)
			symMat.col(t) = arma::conv_to<rmat>::from(symCol);

		symMat = symMat % gamma;
		rmat symProb = arma::sum(symMat, 0);

		nS.col(i) = arma::trans(symProb);
		dS.col(i) = arma::trans(sumRow);
	}

	matNumB += 1.0 / P * nS;
	matDenB += 1.0 / P * dS;

#else
	rmat gamma    = prodMat.submat(0,0,prodMat.n_rows-1,prodMat.n_cols-2);
	rmat symMat   = 0.0 * gamma;
	rmat sumRow   = arma::sum(gamma,1);

	for (i = 0; i < nCB ; i++)
	{
		lmat redO = matO.rows(0,matO.n_elem-2);

		arma::umat symCol = arma::trans(redO == i);
		for(t=0; t< gamma.n_rows; t++)
			symMat.row(t) = arma::conv_to<rmat>::from(symCol);

		symMat = symMat % gamma;
		rmat symProb = arma::sum(symMat, 1);

		nS.row(i) = arma::trans(symProb);
		dS.row(i) = arma::trans(sumRow);
	}

	matNumB += 1.0 / P * nS;
	matDenB += 1.0 / P * dS;

#endif

#if __CY_DEBUG_PRINT__
	matNumB.print("matNumB: ");
	matDenB.print("matDenB: ");
#endif

}
