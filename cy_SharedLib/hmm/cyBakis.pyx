import numpy as np
cimport numpy as np

ctypedef double HMMReal

cdef extern from "bakis.h":
    cppclass Bakis:
        Bakis ( int *szDefs, int *cyO,
        HMMReal *cyA,    HMMReal *cyB,
        HMMReal *cyAlpha, HMMReal *cyBeta,
        HMMReal *cyNumA, HMMReal *cyDenA,
        HMMReal *cyNumB, HMMReal *cyDenB
        );
        void getBakis();
 
cdef class bakis:
    
    cdef Bakis* cobj
    
    def __init__(self,
                 np.ndarray[np.int32_t,ndim=1] szDefs,
                 np.ndarray[np.int32_t,ndim=1] cyO,
                 np.ndarray[np.float64_t,ndim=2] cyA,
                 np.ndarray[np.float64_t,ndim=2] cyB,
                 np.ndarray[np.float64_t,ndim=2] cyAlpha,
                 np.ndarray[np.float64_t,ndim=2] cyBeta,
                 np.ndarray[np.float64_t,ndim=2] cyNumA,
                 np.ndarray[np.float64_t,ndim=2] cyDenA,
                 np.ndarray[np.float64_t,ndim=2] cyNumB,
                 np.ndarray[np.float64_t,ndim=2] cyDenB):
        self.cobj = new Bakis(<int*> szDefs.data, <int*> cyO.data, 
                                     <HMMReal*> cyA.data, <HMMReal*> cyB.data, 
                                     <HMMReal*> cyAlpha.data, <HMMReal*> cyBeta.data,
                                     <HMMReal*> cyNumA.data, <HMMReal*> cyDenA.data,
                                     <HMMReal*> cyNumB.data, <HMMReal*> cyDenB.data)
        if self.cobj == NULL:
            raise MemoryError('Not enough memory.')
    
    def __dealloc__(self):
        del self.cobj
    
    def apply(self):
        self.cobj.getBakis()
        