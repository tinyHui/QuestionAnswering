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


    cdef np.ndarray cc_cdef(self, list sentence, unsigned int EMBEDDING_SIZE):
        # circular correlation
        cdef np.ndarray result
        cdef np.ndarray crt_result
        cdef np.ndarray emb_w
        cdef float v
        cdef unsigned int k
        cdef unsigned int i

        result = np.ones(EMBEDDING_SIZE) # R^1 x EMBEDDING_SIZE
        for emb_w in sentence[0:]:
            if np.sum(emb_w) == 0:
                continue

            crt_result = np.zeros(EMBEDDING_SIZE, dtype=DTYPE)
            for k in range(EMBEDDING_SIZE):
                v = 0
                for i in range(EMBEDDING_SIZE):
                    v += result[i] * emb_w[(k+i) % EMBEDDING_SIZE]
                crt_result[k] = v
            result = crt_result
        return result

    def avg_emb(self, list sentence, unsigned int EMBEDDING_SIZE):
        return self.avg_emb_cdef(sentence, EMBEDDING_SIZE)

    def cc(self, list sentence, unsigned int EMBEDDING_SIZE):
        return self.cc_cdef(sentence, EMBEDDING_SIZE)