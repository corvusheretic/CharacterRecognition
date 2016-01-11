import numpy as np
cimport numpy as np

cdef extern from "cCalcForward.h": 
    int fnCalcForward( int N, double* a, double* b, double* z )  # z = a + b

def CalcForward( N,
    np.ndarray[np.double_t,ndim=1] A,
    np.ndarray[np.double_t,ndim=1] B,
    np.ndarray[np.double_t,ndim=1] Z ):
    """ wrap np arrays to fc( a.data ... ) """
    assert N <= len(A) == len(B) == len(Z)
    fcret = fc( N, <double*> A.data, <double*> B.data, <double*> Z.data )
        # fcret = fc( N, A.data, B.data, Z.data )  grr char*
    return fcret