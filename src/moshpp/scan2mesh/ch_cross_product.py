import numpy as np
import scipy.sparse as sp
from chumpy.ch import Ch, depends_on
from chumpy.utils import col


class CrossProduct(Ch):
    terms = []
    dterms = 'a', 'b'

    def on_changed(self, which):
        if 'a' in which:
            a = self.a.r.reshape((-1, 3))
            self.a1 = a[:, 0]
            self.a2 = a[:, 1]
            self.a3 = a[:, 2]

        if 'b' in which:
            b = self.b.r.reshape((-1, 3))
            self.b1 = b[:, 0]
            self.b2 = b[:, 1]
            self.b3 = b[:, 2]

    def compute_r(self):

        # TODO: maybe use cross directly? is it faster?
        # TODO: check fortran ordering?
        return _call_einsum_matvec(self.Ax, self.b.r)

    def compute_dr_wrt(self, obj):
        if obj not in (self.a, self.b):
            return None

        sz = self.a.r.size
        if not hasattr(self, 'indices') or self.indices.size != sz * 3:
            self.indptr = np.arange(0, (sz + 1) * 3, 3)
            idxs = col(np.arange(0, sz))
            idxs = np.hstack([idxs, idxs, idxs])
            idxs = idxs.reshape((-1, 3, 3))
            idxs = idxs.transpose((0, 2, 1)).ravel()
            self.indices = idxs

        if obj is self.a:
            # m = self.Bx
            # matvec = lambda x : _call_einsum_matvec(m, x)
            # matmat = lambda x : _call_einsum_matmat(m, x)
            # return sp.linalg.LinearOperator((self.a1.size*3, self.a1.size*3), matvec=matvec, matmat=matmat)
            data = self.Bx.ravel()
            result = sp.csc_matrix((data, self.indices, self.indptr), shape=(sz, sz))
            return -result

        elif obj is self.b:
            # m = self.Ax
            # matvec = lambda x : _call_einsum_matvec(m, x)
            # matmat = lambda x : _call_einsum_matmat(m, x)
            # return sp.linalg.LinearOperator((self.a1.size*3, self.a1.size*3), matvec=matvec, matmat=matmat)
            data = self.Ax.ravel()
            result = sp.csc_matrix((data, self.indices, self.indptr), shape=(sz, sz))
            return -result

    @depends_on('a')
    def Ax(self):
        """Compute a stack of skew-symmetric matrices which can be multiplied
        by 'b' to get the cross product. See:

        http://en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication
        """
        #  0         -self.a3   self.a2
        #  self.a3    0        -self.a1
        # -self.a2    self.a1   0
        m = np.zeros((len(self.a1), 3, 3))
        m[:, 0, 1] = -self.a3
        m[:, 0, 2] = +self.a2
        m[:, 1, 0] = +self.a3
        m[:, 1, 2] = -self.a1
        m[:, 2, 0] = -self.a2
        m[:, 2, 1] = +self.a1
        return m

    @depends_on('b')
    def Bx(self):
        """Compute a stack of skew-symmetric matrices which can be multiplied
        by 'a' to get the cross product. See:

        http://en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication
        """
        #  0         self.b3  -self.b2
        # -self.b3   0         self.b1
        #  self.b2  -self.b1   0

        m = np.zeros((len(self.b1), 3, 3))
        m[:, 0, 1] = +self.b3
        m[:, 0, 2] = -self.b2
        m[:, 1, 0] = -self.b3
        m[:, 1, 2] = +self.b1
        m[:, 2, 0] = +self.b2
        m[:, 2, 1] = -self.b1
        return m


def _call_einsum_matvec(m, righthand):
    r = righthand.reshape(m.shape[0], 3)
    return np.einsum('ijk,ik->ij', m, r).ravel()


def _call_einsum_matmat(m, righthand):
    r = righthand.reshape(m.shape[0], 3, -1)
    return np.einsum('ijk,ikm->ijm', m, r).reshape(-1, r.shape[2])
