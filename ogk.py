import numpy as np

class OGK(object):
    def __init__(self, normalize=True):
        if normalize == True:
            self.normalizer = 1.4286
        else:
            self.normalizer = 1.0

    def mad(self, ar):
        return self.normalizer * np.median(np.abs(ar - np.median(ar)))

    def m2(self,data):
        numcols = data.shape[1]
        return np.array([self.mad(data[:, i]) for i in range(numcols)])

    def rescale(self, data):
        numcols = data.shape[1]
        tmp = 1.0/self.m2(data)
        return tmp * data

    def robust_corr(self, x, y):
        t1 = x + y
        t2 = x-y
        m1 = self.mad(t1)
        m2 = self.mad(t2)
        return (m1**2 - m2**2)/4

    def fit(self, data):
        rs = self.rescale(data)
        numcols = data.shape[1]
        res = np.ones((numcols, numcols))
        for i in range(numcols):
            icol = data[:, i]
            for j in range(i+1, numcols):
                jcol = data[:, j]
            corr = self.robust_corr(icol, jcol)
            res[i, j] = corr
            res[j, i] = corr
        _, vecs = np.linalg.eigh(res)
        zz = rs @ vecs
        newmad = self.m2(zz)
        gamma = np.diag(newmad*newmad)
        ae= self.m2(data) * vecs
        location = ae @ newmad
        scatter = ae @ gamma @ ae.T
        return  scatter



