import numpy as np
from scipy.spatial.distance import cosine
import logging
from multiprocessing import Pool


def xcov(set1, set2):
    num_set1, M = set1.shape
    num_set2, N = set2.shape
    # one-to-one paired
    assert(num_set1 == num_set2)

    cov = np.dot(np.transpose(set1), set2)
    logging.info("get XCOV matrix of shape R^%d x %d" % (cov.shape[0], cov.shape[1]))
    return cov


def train(Qs, As):
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
    U, s, V = np.linalg.svd(cov, full_matrices=False)
    return U, V


def find_answer(v_q, As):
    pool = Pool(processes=8)

    # get distance between the question and answer, return with the answer index
    def distance(indx_t):
        indx, t = indx_t
        dist = cosine(v_q, t)
        return indx, dist

    result = pool.map(distance, As)
    best_indx, _ = min(result, key=lambda x: x[1])

    return best_indx
