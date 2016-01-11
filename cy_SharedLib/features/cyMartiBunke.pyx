import numpy as np
cimport numpy as np

cdef extern from "martibunke.h":
    cppclass MartiBunke:
        MartiBunke ( float*, int, int, float*, double );
        void getMartiBunke();

cdef class martiBunke:
    
    cdef MartiBunke* cobj
    
    def __cinit__(self,
                 np.ndarray[np.float32_t,ndim=2] npData,
                 nRows, nCols,
                 np.ndarray[np.float32_t,ndim=2] npMBbuff,
                 thresh):
        self.cobj = new MartiBunke(<float*> npData.data, nRows, nCols, 
                                   <float*> npMBbuff.data, thresh )
        if self.cobj == NULL:
            raise MemoryError('Not enough memory.')
    
    def __dealloc__(self):
        del self.cobj
    
    def apply(self):
        self.cobj.getMartiBunke()
        