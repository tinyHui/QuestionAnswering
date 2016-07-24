from libc.stdlib cimport free
import numpy as np
cimport numpy as np
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

cdef class Utils(object):
    cdef np.ndarray avg_emb_cdef(self, list sentence, unsigned int EMBEDDING_SIZE):
        cdef np.ndarray emb_w
        cdef np.ndarray result

        word_num = 0
        result = np.zeros(EMBEDDING_SIZE, dtype=DTYPE)
        # average of embedding of the words
        for emb_w in sentence:
            if np.sum(emb_w) == 0:
                continue
            # for each token, find its embedding
            # unseen token will automatically take 0 x R^300
            result += emb_w
            word_num += 1
        # all words in answer sentence might not seen
        if word_num > 0:
            result /= word_num
        return result


    cdef np.ndarray cc_calc_cdef(self, np.ndarray a, np.ndarray b, unsigned int EMBEDDING_SIZE):
        # circular correlation
        cdef unsigned int k
        cdef unsigned int i
        cdef float v

        # if vec is all 0, we skip it; if vec is all 1, it does not affect the calculation
        cdef bint a_useless = np.sum(a) == 0 and np.all(a == 1)
        cdef bint b_useless = np.sum(b) == 0 and np.all(b == 1)

        cdef np.ndarray result = np.ones(EMBEDDING_SIZE) # R^1 x EMBEDDING_SIZE
        if a_useless or b_useless:
            # value of a or b does not affect the result
            if a_useless and b_useless:
                return result
            elif a_useless:
                return b
            elif b_useless:
                return a

        # a and b is not useless value
        for k in range(EMBEDDING_SIZE):
            v = 0
            for i in range(EMBEDDING_SIZE):
                v += a[i] * b[(k+i) % EMBEDDING_SIZE]
            result[k] = v
        return result


    def avg_emb(self, list sentence, unsigned int EMBEDDING_SIZE):
        return self.avg_emb_cdef(sentence, EMBEDDING_SIZE)


    def cc(self, list struct, unsigned int EMBEDDING_SIZE):
        result = np.ones(EMBEDDING_SIZE)
        for emb_w in struct:
            if isinstance(emb_w, list):
                part_result = self.cc(emb_w, EMBEDDING_SIZE)
                result = self.cc_calc_cdef(result, part_result, EMBEDDING_SIZE)
            else:
                result = self.cc_calc_cdef(result, emb_w, EMBEDDING_SIZE)

        return result
