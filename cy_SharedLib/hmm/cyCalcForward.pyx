import numpy as np
cimport numpy as np

ctypedef double HMMReal

cdef extern from "calcforward.h":
    cppclass CalcForward:
        CalcForward ( int *, int *,
            HMMReal *, HMMReal *, HMMReal *,
            HMMReal *, HMMReal * );
        void getCalcForward();
 
cdef class calcForward:
    
    cdef CalcForward* cobj
    
    def __cinit__(self,
                 np.ndarray[np.int32_t,ndim=1] szDefs,
                 np.ndarray[np.int32_t,ndim=1] cyO,
                 np.ndarray[np.float64_t,ndim=2] cyA,
                 np.ndarray[np.float64_t,ndim=2] cyB,
                 np.ndarray[np.float64_t,ndim=1] cyPI,
                 np.ndarray[np.float64_t,ndim=1] cySF,
                 np.ndarray[np.float64_t,ndim=2] cyAlpha):
        self.cobj = new CalcForward(<int*> szDefs.data, <int*> cyO.data, 
                                   <HMMReal*> cyA.data, <HMMReal*> cyB.data,
                                   <HMMReal*> cyPI.data, <HMMReal*> cySF.data,
                                   <HMMReal*> cyAlpha.data )
        if self.cobj == NULL:
            raise MemoryError('Not enough memory.')
    
    def __dealloc__(self):
        del self.cobj
    
    def apply(self):
        self.cobj.getCalcForward()
        