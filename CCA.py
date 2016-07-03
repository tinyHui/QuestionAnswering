import numpy as np
from scipy.linalg import sqrtm as dense_sqrt
from scipy.linalg import inv as dense_inv
from scipy.sparse.linalg import inv as sparse_inv
from scipy.sparse import diags
from scipy.sparse.linalg import svds
from scipy.spatial.distance import cosine
import logging


def train(Qs, As, sample_num=0, full_svd=True, k=0, sparse=False):
    '''
    train use sparse matrix
    params q: sentence embedding for question set
    params a: sentence embedding for answer set
    '''

    logging.info("calculating C_AA")
    c_qq = Qs.T.dot(Qs)
    logging.info("calculating C_BB")
    c_aa = As.T.dot(As)
    logging.info("calculating C_AB")
    c_qa = Qs.T.dot(As) / sample_num

    if sparse:
        # sparse
        logging.info("keep only diagonal")
        c_qq = diags(c_qq.diagonal(), 0)
        c_aa = diags(c_aa.diagonal(), 0)

        logging.info("doing square root and invert for C_AA")
        c_qq_sqrt = sparse_inv(c_qq.sqrt()) / sample_num
        logging.info("doing square root and invert for C_BB")
        c_aa_sqrt = sparse_inv(c_aa.sqrt()) / sample_num
        logging.info("C_AA * C_AB * C_BB")
        result = c_qq_sqrt.dot(c_qa).dot(c_aa_sqrt)
    else:
        # dense
        logging.info("keep only diagonal")
        c_qq = np.diag(np.diag(c_qq))
        c_aa = np.diag(np.diag(c_aa))

        logging.info("doing square root and invert for C_AA")
        c_qq_sqrt = dense_inv(dense_sqrt(c_qq)) / sample_num
        logging.info("doing square root and invert for C_BB")
        c_aa_sqrt = dense_inv(dense_sqrt(c_aa)) / sample_num
        logging.info("C_AA * C_AB * C_BB")
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
def distance(proj_q, proj_a):
    dist = cosine(proj_q, proj_a)
    return dist
