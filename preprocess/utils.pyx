from libc.stdlib cimport free
import numpy as np
cimport numpy as np
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t


def avg_emb(list sentence, np.ndarray EMBEDDING_SIZE):
    cdef unsigned int w
    cdef np.ndarray result

    result = np.zeros(EMBEDDING_SIZE, dtype=DTYPE)
    # average of embedding of the words
    for w in sentence:
        # for each token, find its embedding
        # unseen token will automatically take 0 x R^300
        result += w
    result /= len(sentence)
    return result


def cc(list sentence, np.ndarray EMBEDDING_SIZE):
    # circular correlation
    cdef np.ndarray result
    cdef np.ndarray crt_result
    cdef np.ndarray emb_w
    cdef float v
    cdef unsigned int k
    cdef unsigned int i

    result = sentence[0] # R^1 x EMBEDDING_SIZE
    for emb_w in sentence[1:]:
        crt_result = np.zeros(EMBEDDING_SIZE, dtype=DTYPE)
        for k in range(EMBEDDING_SIZE):
            v = 0
            for i in range(EMBEDDING_SIZE):
                v += result[i] * emb_w[(k+i) % EMBEDDING_SIZE]
            crt_result[k] = v
        result = crt_result
    return result