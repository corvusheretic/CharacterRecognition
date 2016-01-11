import numpy as np
cimport numpy as np

ctypedef double HMMReal

cdef extern from "calcbackward.h":
    cppclass CalcBackward:
        CalcBackward ( int *,  int *,
            HMMReal *, HMMReal *, 
            HMMReal *, HMMReal *);
        void getCalcBackward();
 
cdef class calcBackward:
    
    cdef CalcBackward* cobj
    
    def __cinit__(self,
                 np.ndarray[np.int32_t,ndim=1] szDefs,
                 np.ndarray[np.int32_t,ndim=1] cyO,
                 np.ndarray[np.float64_t,ndim=2] cyA,
                 np.ndarray[np.float64_t,ndim=2] cyB,
                 np.ndarray[np.float64_t,ndim=1] cySF,
                 np.ndarray[np.float64_t,ndim=2] cyBeta):
        self.cobj = new CalcBackward(<int*> szDefs.data, <int*> cyO.data, 
                                     <HMMReal*> cyA.data, <HMMReal*> cyB.data,
                                     <HMMReal*> cySF.data, <HMMReal*> cyBeta.data )
        if self.cobj == NULL:
            raise MemoryError('Not enough memory.')
    
    def __dealloc__(self):
        del self.cobj
    
    def apply(self):
        self.cobj.getCalcBackward()
        