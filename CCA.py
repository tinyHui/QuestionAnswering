import numpy as np
from scipy.spatial.distance import cosine
import logging

def xcov(set1, set2):
    num_set1, M = set1.shape
    num_set2, N = set2.shape
    # one-to-one paired
    assert(num_set1 == num_set2)

    cov = np.dot(np.transpose(set1), set2)
    logging.info("get XCOV matrix of shape %s" % cov.shape)
    return cov


class CCA(object):
    def __init__(self):
        self.U = None
        self.V = None

    def train(self, Qs, As):
        '''
        params q: sentence embedding for question set
        params a: sentence embedding for answer set
        '''
        if isinstance(Qs, list):
            Qs = np.asarray(Qs, dtype="float64")
        if isinstance(As, list):
            As = np.asarray(As, dtype="float64")

        logging.info("computing cross-covariance matrix")
        cov = xcov(Qs, As)
        logging.info("decomposition using SVD")
        self.U, s, self.V = np.linalg.svd(cov, full_matrices=False)

    def find_answer(self, v_q, As):
        assert self.U is not None and \
               self.V is not None
        if isinstance(v_q, list):
            v_q = np.asarray(v_q, dtype="float64")
        if isinstance(As, list):
            As = np.asarray(As, dtype="float64")

        v_q_proj = np.dot(v_q, self.U)

        best_distance = np.inf
        best_indx = 0
        for i, v_a in enumerate(As):
            v_a_proj = np.dot(v_a, self.V.T)
            s = cosine(v_q_proj, v_a_proj)
            if s <= best_distance:
                best_distance = s
                best_indx = i

        return best_indx
