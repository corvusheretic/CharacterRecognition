/*
 * cy_sharedlib.h
 *
 *  Created on: Dec 30, 2015
 *      Author: kalyan
 */

#ifndef CY_SHAREDLIB_H_
#define CY_SHAREDLIB_H_

#include "common/globaldefs.h"

#include "hmm/bakis.h"
#include "hmm/baumwelch.h"
#include "hmm/calcbackward.h"
#include "hmm/calcforward.h"
int mainBK();
int mainBW();
int mainCB();
int mainCF();


#include "features/martibunke.h"
int mainMB();


#endif /* CY_SHAREDLIB_H_ */
