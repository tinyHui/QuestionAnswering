from libc.stdlib cimport free
import numpy as np
cimport numpy as np
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t


def avg_emb(sentence, embedding_dict, result):
    # average of embedding of the words
    for w in sentence:
        # for each token, find its embedding
        # unseen token will automatically take 0 x R^300
        result += embedding_dict[w]
    result /= len(sentence)
    return result


def cc(sentence, embedding_dict, d):
    # circular correlation
    cdef unsigned int w
    cdef np.ndarray result
    cdef np.ndarray crt_result
    cdef np.ndarray emb_w
    cdef float v

    w = sentence[0]
    result = embedding_dict[w] # R^1 x 300
    for w in sentence[1:]:
        emb_w = embedding_dict[w]     # R^1 x 300
        crt_result = np.zeros(d, dtype=DTYPE)
        for k in range(d):
            v = 0
            for i in range(d):
                v += result[i] * emb_w[(k+i) % d]
            crt_result[k] = v
        result = crt_result
    return result