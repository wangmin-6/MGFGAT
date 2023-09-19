import numpy
import numpy as np
from scipy.linalg import eigh as largest_eigh
from sklearn.neighbors import kneighbors_graph


def prox_wtnn(Y, C):
    # weighted tensor nuclear norm
    # min_X ||X||_w* + 0.5||X - Y||_F^2
    n1, n2, n3 = np.shape(Y)
    X = np.zeros((n1, n2, n3), dtype=complex)
    Y = np.fft.fftn(Y)
    eps = 1e-6
    for i in range(n3):
        U, S, V = np.linalg.svd(Y[:, :, i], full_matrices=False)
        temp = np.power(S - eps, 2) - 4*(C - eps*S)
        ind = np.where(temp > 0)
        ind = np.array(ind)
        r = np.max(ind.shape)
        if np.min(ind.shape) == 0:
            r = 0
        if r >= 1:
            temp2 = S[ind] - eps + np.sqrt(temp[ind])
            S = temp2.reshape(temp2.size, )
            X[:, :, i] = np.dot(np.dot(U[:, 0:r], np.diag(S)), V[:, 0:r].T)
    newX = np.fft.ifftn(X)
    return np.real(newX)
# import seaborn

def constructW(Dist, n_neighbors):

    neighbors_graph = kneighbors_graph(
        Dist, n_neighbors, mode='distance', include_self=False)
    W = 0.5 * (neighbors_graph + neighbors_graph.T)

    return W


def cgl(A, n_feature,k ,lambda_1 = 1, rho=1, n_iter=100):
    # consensus graph learning
    # min_H, Z 0.5||A - H'H||_F^2 + 0.5||Z - hatHhatH'||_F^2 + ||Z||_w*
    # s.t. H'H = I_k
    n_sample, n_sample, n_view = np.shape(A)

    # matrix initial
    H = np.zeros((n_sample, n_feature, n_view))
    HH = np.zeros((n_sample, n_sample, n_view))
    hatH = np.zeros((n_sample, n_feature, n_view))
    hatHH = np.zeros((n_sample, n_sample, n_view))
    Q = np.zeros((n_sample, n_sample, n_view))
    Z = np.zeros((n_sample, n_sample, n_view))

    # repeat
    for iter in range(n_iter):
        # update H
        temp = np.zeros((n_sample, n_sample, n_view))
        G = np.zeros((n_sample, n_sample, n_view))
        for view in range(n_view):
            temp[:, :, view] = np.dot(np.dot(Q[:, :, view], 0.5*(Z[:, :, view] + Z[:, :, view].T) - 0.5*hatHH[:, :, view]), Q[:, :, view])
            G[:, :, view] = lambda_1*A[:, :, view] + temp[:, :, view]
            _, H[:, :, view] = largest_eigh(G[:, :, view], eigvals=[n_sample-n_feature, n_sample-1])
            HH[:, :, view] = np.dot(H[:, :, view], H[:, :, view].T)
            Q[:, :, view] = np.diag(1/np.sqrt(np.diag(HH[:, :, view])))
            hatH[:, :, view] = np.dot(Q[:, :, view], H[:, :, view])
            hatHH[:, :, view] = np.dot(hatH[:, :, view], hatH[:, :, view].T)
        # update Z
        hatHH2 = hatHH.transpose((0, 2, 1))
        Z2 = prox_wtnn(hatHH2, rho)
        Z = Z2.transpose((0, 2, 1))
    # compute similarity matrix
    Dist = np.zeros((n_sample, n_sample))
    for view in range(n_view):
        Dist += hatHH[:, :, view]
    W = constructW(1 - Dist, k)
    W = W.todense()
    W = numpy.array(W)
    W = (W - W.min()) / (W.max() - W.min())
    return W


