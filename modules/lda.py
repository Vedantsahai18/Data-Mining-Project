import numpy as np
from numpy.linalg import eig, pinv, norm

def getGaussianData(m, C, n=1000):
    x = np.random.multivariate_normal(m, C, n)
    return x

def getT(means):
    M = np.array(means).astype(np.float64).T
    m = M.mean(axis=1, keepdims=True)
    dists = norm(M - m, 2, axis=0)
    dist = np.max(dists)
    return dist

def getLine(means, w, T=5, N=2):
    if len(means) != 1:
        T = getT(means)*T
    m = np.array(means).astype(np.float64).mean(axis=0)
    p = np.outer(np.ones(N), m).T
    t = np.linspace(0, T, N)
    v = np.outer(np.ones(N), w).T
    line1 = p + v * (t - T)
    line2 = p + v * (T - t)
    line = np.concatenate((line1[:, :-1], line2[:, ::-1]), axis=1)
    line = line[:, [0, -1]]
    return line

def topK_Eig(M, k):
    evals, U = eig(M)
    idx = np.argsort(evals)[::-1][:k]
    evals = evals[idx]
    U = U[:, idx]
    return evals, U

def computeDiscriminant(means, covarianceMatrices):
    M = np.array(means).astype(np.float64).T
    m0 = M.mean(axis=1, keepdims=True)
    M -= m0
    S_W = sum(covarianceMatrices).astype(np.float64)
    S_B = M @ M.T
    evals, U = topK_Eig(pinv(S_W) @ S_B, 1)
    w = U[:, 0]
    return w

def lda(data):
    means = data["means"]
    means = [np.array(meanVec).astype(np.float64) for meanVec in means]
    covMats = data["covarianceMatrices"]
    covMats = [np.array(covMat).astype(np.float64) for covMat in covMats]

    if len(means) != len(covMats) or len(means) == 0:
        return {
            'points': [],
            'line': []
        }

    points = []
    for label, pair in enumerate(zip(means, covMats)):
        m, C = pair
        pts = getGaussianData(m, C, n=100)
        for idx in range(pts.shape[0]):
            points.append({
                'x': pts[idx, 0],
                'y': pts[idx, 1],
                'label': label
            })
    
    w = computeDiscriminant(means, covMats)
    line = getLine(means, w, T=5, N=2)
    line = [{'x': line[0, i], 'y': line[1, i]} for i in range(line.shape[1])]

    output_data = {
            'points': points,
            'line': line
    }

    return output_data
