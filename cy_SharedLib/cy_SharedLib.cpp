//============================================================================
// Name        : cy_SharedLib.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include "cy_sharedlib.h"
using namespace std;

int main(int argc,char *argv[]){

	if(argc == 2) {
		if(strcmp(argv[1],"mb") == 0)
			mainMB();
		else if(strcmp(argv[1],"bk") == 0)
			mainBK();
		else if(strcmp(argv[1],"bw") == 0)
			mainBW();
		else if(strcmp(argv[1],"cf") == 0)
			mainCF();
		else if(strcmp(argv[1],"cb") == 0)
			mainCB();
	}
	else {
		cout << "\t <<<< Enter one of the options below for test run. " << endl;
		cout << "\t\t >> mb for Marti-Bunke" << endl;
		cout << "\t\t >> bk for Bakis" << endl;
		cout << "\t\t >> bw for Baum-Welch" << endl;
		cout << "\t\t >> cf for CalcForward" << endl;
		cout << "\t\t >> cb for CalcBackward" << endl;
	}

	return 0;
}
