import numpy as np
from scipy.spatial.distance import cosine
import logging


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

    print(Qs.shape, As.shape)
    logging.info("computing cross-covariance matrix")
    cov = xcov(Qs, As)
    logging.info("decomposition using SVD")
    U, s, V = np.linalg.svd(cov, full_matrices=False)
    return U, s, V


def find_answer(v_q, As, U, V):
    assert U is not None and \
           V is not None
    if isinstance(v_q, list):
        v_q = np.asarray(v_q, dtype="float64")
    if isinstance(As, list):
        As = np.asarray(As, dtype="float64")

    v_q_proj = np.dot(v_q, U)

    best_distance = np.inf
    best_indx = 0
    for i, v_a in enumerate(As):
        v_a_proj = np.dot(v_a, V.T)
        s = cosine(v_q_proj, v_a_proj)
        if s <= best_distance:
            best_distance = s
            best_indx = i

    return best_indx


