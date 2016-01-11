/*
 * cyMain.cpp
 *
 *  Created on: Dec 6, 2015
 *      Author: kalyan
 */


#include <iostream>
//#include <opencv2/opencv.hpp>

#include "calcforward.h"
#include "calcbackward.h"
#include "baumwelch.h"

//using namespace cv;
using namespace std;

#define N 6
#define M 5
#define nLabels 7
#define K nLabels+2

int mainBW()
{
	int cyO[M] = {5,0, 3, 3, 7};
	HMMReal cyA[N][N] = {{ 0.11309484, 0.17242173, 0.11681411, 0.23805921, 0.25725022, 0.10235989},
	 { 0.26635647, 0.17793372, 0.19110466, 0.31139429, 0.02389834, 0.02931252},
	 { 0.00472527, 0.19459263, 0.18186399, 0.20333163, 0.22871412, 0.18677235},
	 { 0.14943185, 0.25274353, 0.0382985,  0.20721313, 0.0464193,  0.30589369},
	 { 0.17395618, 0.138226, 0.08818862, 0.25808789, 0.152056, 0.18948532},
	 { 0.00538232, 0.17692104, 0.17533418, 0.1767201,  0.27033565, 0.19530671}};
	HMMReal cyB[N][K] = {{ 0.10136868, 0.12322775, 0.19670764, 0.01698148, 0.18800494, 0.18909647
	,  0.05932054, 0.03635272, 0.08893978},
	 { 0.09726151, 0.15247885, 0.11728838, 0.26430543, 0.02728826, 0.05585666
	,  0.04313649, 0.17465059, 0.06773383},
	 { 0.14752495, 0.07732798, 0.0502926,  0.03491896, 0.20764047, 0.04371641
	,  0.06219201, 0.11665216, 0.25973447},
	 { 0.02007754, 0.17326109, 0.01987018, 0.20190162, 0.09690257, 0.20196398
	,  0.12506335, 0.15285684, 0.00810282},
	 { 0.09842451, 0.04183167, 0.10306484, 0.04132047, 0.11066679, 0.14417479
	,  0.02232507, 0.24099913, 0.19719271},
	 { 0.06285983, 0.12393589, 0.02225063, 0.13641798, 0.22011195, 0.07545585
	,  0.158082,   0.03121748, 0.16966838}};

	HMMReal cyPI[N] = { 0.28940609, 0.18319136, 0.58651293, 0.02010755, 0.82894003, 0.00469548};
	HMMReal cyAlpha[M][N];
	HMMReal cySF[M];
	HMMReal cyBeta[M][N];

	int sizeDefines[] = {0,0,0,0,0,0,0};

	memset(&cyAlpha[0][0],0,sizeof(cyAlpha));
	memset(&cyBeta[0][0],0,sizeof(cyBeta));

	sizeDefines[0] = N;
	sizeDefines[1] = M;
	sizeDefines[2] = N;
	sizeDefines[3] = N;
	sizeDefines[4] = N;
	sizeDefines[5] = K;
	sizeDefines[6] = N;

	CalcForward cf( &sizeDefines[0], &cyO[0],
				&cyA[0][0], &cyB[0][0], &cyPI[0],
				&cySF[0], &cyAlpha[0][0] );
	cf.getCalcForward();

#if __CY_DEBUG_PRINT__
    std::cout << "Alpha in Main";
    for (int i=0; i < M; i++)
    {
    	std::cout << "" <<std::endl;
    	for (int j=0; j < N; j++)
                std::cout << cyAlpha[i][j] << ' ';
    }
    std::cout << "" <<std::endl;

    std::cout << "SF in Main";
    for (int i=0; i < M; i++)
    	std::cout << cySF[i] << ' ';
    std::cout << "" <<std::endl;
#endif

	CalcBackward cb( &sizeDefines[0], &cyO[0],
			&cyA[0][0], &cyB[0][0],
			&cySF[0], &cyBeta[0][0] );
	cb.getCalcBackward();

#if __CY_DEBUG_PRINT__
    std::cout << "Beta in Main";
    for (int i=0; i < M; i++)
    {
    	std::cout << "" <<std::endl;
    	for (int j=0; j < N; j++)
                std::cout << cyBeta[i][j] << ' ';
    }
    std::cout << "" <<std::endl;
#endif

    BaumWelch bw( &sizeDefines[0], &cyO[0],
			&cyA[0][0], &cyB[0][0],
			&cyPI[0], &cySF[0],
			&cyAlpha[0][0], &cyBeta[0][0] );
    bw.getBaumWelch();


#if __CY_DEBUG_PRINT__

    std::cout << "pi in Main";
    for (int i=0; i < N; i++)
    	std::cout << cyPI[i] << ' ';
    std::cout << "" <<std::endl;

    std::cout << "matA in Main";
    for (int i=0; i < N; i++)
    {
    	std::cout << "" <<std::endl;
    	for (int j=0; j < N; j++)
                std::cout << cyA[i][j] << ' ';
    }
    std::cout << "" <<std::endl;

    std::cout << "matB in Main";
    for (int i=0; i < N; i++)
    {
    	std::cout << "" <<std::endl;
    	for (int j=0; j < K; j++)
                std::cout << cyB[i][j] << ' ';
    }
    std::cout << "" <<std::endl;

#endif

    return 0;
}

