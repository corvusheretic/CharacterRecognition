import numpy as np
cimport numpy as np

ctypedef double HMMReal

cdef extern from "baumwelch.h":
    cppclass BaumWelch:
        BaumWelch ( int *szDefs, int *cyO,
            HMMReal *cyA,    HMMReal *cyB,
            HMMReal *cyPI,    HMMReal *cySF,
            HMMReal *cyAlpha, HMMReal *cyBeta );
        void getBaumWelch();
 
cdef class fnBaumWelch:
    
    cdef BaumWelch* cobj
    
    def __cinit__(self,
                 np.ndarray[np.int32_t,ndim=1] szDefs,
                 np.ndarray[np.int32_t,ndim=1] cyO,
                 np.ndarray[np.float64_t,ndim=2] cyA,
                 np.ndarray[np.float64_t,ndim=2] cyB,
                 np.ndarray[np.float64_t,ndim=1] cyPI,
                 np.ndarray[np.float64_t,ndim=1] cySF,
                 np.ndarray[np.float64_t,ndim=2] cyAlpha,
                 np.ndarray[np.float64_t,ndim=2] cyBeta):
        self.cobj = new BaumWelch(<int*> szDefs.data, <int*> cyO.data, 
                                     <HMMReal*> cyA.data, <HMMReal*> cyB.data,
                                     <HMMReal*> cyPI.data,<HMMReal*> cySF.data, 
                                     <HMMReal*> cyAlpha.data, <HMMReal*> cyBeta.data )
        if self.cobj == NULL:
            raise MemoryError('Not enough memory.')
    
    def __dealloc__(self):
        del self.cobj
    
    def apply(self):
        self.cobj.getBaumWelch()
        