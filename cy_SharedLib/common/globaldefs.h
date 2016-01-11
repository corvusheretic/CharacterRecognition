/*
 * globaldefs.h
 *
 *  Created on: Dec 22, 2015
 *      Author: kalyan
 */

#ifndef GLOBALDEFS_H_
#define GLOBALDEFS_H_

#define ARMA_NO_DEBUG
#include <armadillo>

#define __CY_DEBUG_PRINT__ 0
#define __ROW_MAJOR_ACCESS__ 0

typedef double HMMReal;
typedef arma::Mat <int> lmat;
typedef arma::Mat <HMMReal> rmat;
typedef arma::Cube <HMMReal> rcube;


#endif /* GLOBALDEFS_H_ */
