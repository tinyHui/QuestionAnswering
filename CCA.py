import numpy as np
from scipy.sparse.linalg import svds
from scipy.spatial.distance import cosine
import logging
from multiprocessing import Pool
from functools import partial


def train(Qs, As, full_svd=True, k=0):
    '''
    params q: sentence embedding for question set
    params a: sentence embedding for answer set
    '''
    if isinstance(Qs, list):
        Qs = np.asarray(Qs, dtype="float64")
    if isinstance(As, list):
        As = np.asarray(As, dtype="float64")

    logging.info("computing CCA")
    sample_num = Qs.shape[0]
    c_qq_sqrt = np.power(Qs.T.dot(Qs), -0.5) / sample_num
    c_qa = Qs.T.dot(As) / sample_num
    c_aa_sqrt = np.power(As.T.dot(As), -0.5) / sample_num
    # keep only diagonal
    c_qq_sqrt = np.diag(np.diag(c_qq_sqrt))
    c_aa_sqrt = np.diag(np.diag(c_aa_sqrt))
    # get result
    result = c_qq_sqrt.dot(c_qa).dot(c_aa_sqrt)
    logging.info("decompose on cross covariant matrix \in R^%d x %d" % (result.shape[0], result.shape[1]))
    if full_svd:
        U, s, V = np.linalg.svd(result, full_matrices=False)
    else:
        U, s, V = svds(result, k=k)
    Q_k = c_qq_sqrt.dot(U)
    A_k = c_aa_sqrt.dot(V.T)
    return Q_k, A_k


# get distance between the question and answer, return with the answer index
def distance(indx_a, proj_q):
    indx, proj_a = indx_a
    dist = cosine(proj_q, proj_a)
    return indx, dist


def find_answer(proj_q, proj_As):
    with Pool(processes=8) as pool:
        result = pool.map(partial(distance, proj_q=proj_q), enumerate(proj_As))
    best_indx, _ = min(result, key=lambda x: x[1])

    return best_indx
