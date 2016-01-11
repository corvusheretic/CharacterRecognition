/*
 * cyMain.cpp
 *
 *  Created on: Dec 6, 2015
 *      Author: kalyan
 */


#include <iostream>

#include "bakis.h"
//#include <opencv2/opencv.hpp>

#include "calcforward.h"
#include "calcbackward.h"

//using namespace cv;
using namespace std;

#define N 6
#define M 5
#define nLabels 7
#define K nLabels+2
#define nSlices 4

int mainBK()
{
	int cyO[nSlices][M] = {	{5, 0, 3, 3, 7},
							{3, 5, 2, 4, 7},
							{6, 8, 8, 1, 6},
							{7, 7, 8, 1, 5}
				  	  	  };
	HMMReal cyA[N][N] = {{ 0.25432872,  0.10263331,  0.19716889,  0.11201624,  0.29115922,  0.04269363},
			 { 0.2140649 ,  0.11652033,  0.19704561,  0.12805148,  0.16702265,  0.17729504},
			 { 0.22014109,  0.20325413,  0.28693606,  0.04005811,  0.17913293,  0.07047768},
			 { 0.41288003,  0.12132869,  0.07575995,  0.18160951,  0.08385974,  0.12456208},
			 { 0.11524154,  0.26913273,  0.13416405,  0.18280049,  0.26905822,  0.02960297},
			 { 0.27625735,  0.18605181,  0.04868487,  0.10202232,  0.21383855,  0.1731451 }};
	HMMReal cyB[N][K] = {{ 0.06207855,  0.00733861,  0.12113567,  0.18314316,  0.12467196,  0.12128555, 0.19008546,  0.11112362,  0.07913742},
	 { 0.09823663,  0.12903276,  0.06994924,  0.13963712,  0.06564355,  0.16107529, 0.19650969,  0.13710712,  0.00280859},
	 { 0.13467405,  0.14566114,  0.21015749,  0.18988619,  0.11019284,  0.01204683, 0.09755129,  0.00432181,  0.09550835},
	 { 0.16451447,  0.06036608,  0.08076257,  0.11565564,  0.14786952,  0.15421097, 0.0364137 ,  0.09491936,  0.14528768},
	 { 0.10082707,  0.18160339,  0.1824819 ,  0.01646464,  0.05501622,  0.00185357, 0.16686849,  0.1282056 ,  0.16667912},
	 { 0.07654712,  0.11503046,  0.15984915,  0.04769278,  0.10693056,  0.04234385, 0.16470601,  0.20348668,  0.08341339}};

	HMMReal cyPI[N] = { 0.10799783,  0.18846096,  0.09872821,  0.22333919,  0.15925224,  0.22222156};
	HMMReal cyAlpha[M][N];
	HMMReal cySF[M];
	HMMReal cyBeta[M][N];

	HMMReal cyNumA[N][N];
	HMMReal cyDenA[N][N];
	HMMReal cyNumB[N][K];
	HMMReal cyDenB[N][K];

	int sizeDefines[] = {0,0,0,0,0,0,0};

	memset(&cyAlpha[0][0],0,sizeof(cyAlpha));
	memset(&cyBeta[0][0],0,sizeof(cyBeta));

	memset(&cyNumA[0][0],0,sizeof(cyNumA));
	memset(&cyDenA[0][0],0,sizeof(cyDenA));

	memset(&cyNumB[0][0],0,sizeof(cyNumB));
	memset(&cyDenB[0][0],0,sizeof(cyDenB));

	sizeDefines[0] = N;
	sizeDefines[1] = M;
	sizeDefines[2] = N;
	sizeDefines[3] = N;
	sizeDefines[4] = N;
	sizeDefines[5] = K;
	sizeDefines[6] = N;

	for(int k=0; k < 1; k++)
	{
		CalcForward cf( &sizeDefines[0], &cyO[k][0],
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

		CalcBackward cb( &sizeDefines[0], &cyO[k][0],
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

		Bakis bk( &sizeDefines[0], &cyO[k][0],
				&cyA[0][0], &cyB[0][0],
				&cyAlpha[0][0], &cyBeta[0][0],
				&cyNumA[0][0], &cyDenA[0][0],
				&cyNumB[0][0], &cyDenB[0][0]
				);
		bk.getBakis();

#if __CY_DEBUG_PRINT__

		std::cout << "NumA in Main";
		for (int i=0; i < N; i++)
		{
			std::cout << "" <<std::endl;
			for (int j=0; j < N; j++)
					std::cout << cyNumA[i][j] << ' ';
		}
		std::cout << "" <<std::endl;

		std::cout << "DenA in Main";
		for (int i=0; i < N; i++)
		{
			std::cout << "" <<std::endl;
			for (int j=0; j < N; j++)
					std::cout << cyDenA[i][j] << ' ';
		}
		std::cout << "" <<std::endl;

		std::cout << "NumB in Main";
		for (int i=0; i < N; i++)
		{
			std::cout << "" <<std::endl;
			for (int j=0; j < K; j++)
					std::cout << cyNumB[i][j] << ' ';
		}
		std::cout << "" <<std::endl;

		std::cout << "DenB in Main";
		for (int i=0; i < N; i++)
		{
			std::cout << "" <<std::endl;
			for (int j=0; j < K; j++)
					std::cout << cyDenB[i][j] << ' ';
		}
		std::cout << "" <<std::endl;

#endif

	}

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

