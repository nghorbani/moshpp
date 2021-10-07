import chumpy as ch
import numpy as np
import scipy.sparse as sp
from chumpy.ch import Ch, MatVecMult
from chumpy.utils import row, col

from moshpp.scan2mesh.ch_cross_product import CrossProduct


class TriEdges(Ch):
    terms = 'f', 'cplus', 'cminus'
    dterms = 'v'

    def compute_r(self):
        cplus = self.cplus
        cminus = self.cminus
        return _edges_for(self.v.r, self.f, cplus, cminus)

    def compute_dr_wrt(self, wrt):
        if wrt is not self.v:
            return None

        cplus = self.cplus
        cminus = self.cminus
        vplus = self.f[:, cplus]
        vminus = self.f[:, cminus]
        vplus3 = row(np.hstack([col(vplus * 3), col(vplus * 3 + 1), col(vplus * 3 + 2)]))
        vminus3 = row(np.hstack([col(vminus * 3), col(vminus * 3 + 1), col(vminus * 3 + 2)]))

        IS = row(np.arange(0, vplus3.size))
        ones = np.ones(vplus3.size)
        shape = (self.f.size, self.v.r.size)
        return sp.csc_matrix((ones, np.vstack([IS, vplus3])), shape=shape) - sp.csc_matrix(
            (ones, np.vstack([IS, vminus3])), shape=shape)


def _edges_for(v, f, cplus, cminus):
    return (
            v.reshape(-1, 3)[f[:, cplus], :] -
            v.reshape(-1, 3)[f[:, cminus], :]).ravel()


class NormalizedNx3(Ch):
    dterms = 'v'

    def on_changed(self, which):
        if 'v' in which:
            self.ss = np.sum(self.v.r.reshape(-1, 3) ** 2, axis=1)
            self.ss[self.ss == 0] = 1e-10
            self.s = np.sqrt(self.ss)
            self.s_inv = 1. / self.s

    def compute_r(self):
        return (self.v.r.reshape(-1, 3) / col(self.s)).reshape(self.v.r.shape)

    def compute_dr_wrt(self, wrt):
        if wrt is not self.v:
            return None

        v = self.v.r.reshape(-1, 3)
        blocks = -np.einsum('ij,ik->ijk', v, v) * (self.ss ** (-3. / 2.)).reshape((-1, 1, 1))
        for i in range(3):
            blocks[:, i, i] += self.s_inv

        if True:
            data = blocks.ravel()
            indptr = np.arange(0, (self.v.r.size + 1) * 3, 3)
            indices = col(np.arange(0, self.v.r.size))
            indices = np.hstack([indices, indices, indices])
            indices = indices.reshape((-1, 3, 3))
            indices = indices.transpose((0, 2, 1)).ravel()
            result = sp.csc_matrix((data, indices, indptr), shape=(self.v.r.size, self.v.r.size))
            return result
        else:
            matvec = lambda x: np.einsum('ijk,ik->ij', blocks, x.reshape((blocks.shape[0], 3))).ravel()
            return sp.linalg.LinearOperator((self.v.r.size, self.v.r.size), matvec=matvec)


def TriNormals(v, f):
    return NormalizedNx3(TriNormalsScaled(v, f))


def TriNormalsScaled(v, f):
    return CrossProduct(TriEdges(f, 1, 0, v), TriEdges(f, 2, 0, v))


class VertNormals(Ch):
    """If normalized==True, normals are normalized; otherwise they'll be about as long as neighboring edges."""

    dterms = 'v'
    terms = 'f', 'normalized'
    term_order = 'v', 'f', 'normalized'

    def on_changed(self, which):

        if not hasattr(self, 'normalized'):
            self.normalized = True

        if hasattr(self, 'v') and hasattr(self, 'f'):
            if 'f' not in which and hasattr(self, 'faces_by_vertex') and self.faces_by_vertex.shape[0] / 3 == \
                    self.v.shape[0]:
                self.tns.v = self.v
            else:  # change in f or in size of v. shouldn't happen often.
                f = self.f

                IS = f.ravel()
                JS = np.array([range(f.shape[0])] * 3).T.ravel()
                data = np.ones(len(JS))

                IS = np.concatenate((IS * 3, IS * 3 + 1, IS * 3 + 2))
                JS = np.concatenate((JS * 3, JS * 3 + 1, JS * 3 + 2))
                data = np.concatenate((data, data, data))

                sz = self.v.size
                self.faces_by_vertex = sp.csc_matrix((data, (IS, JS)), shape=(sz, f.size))

                self.tns = Ch(lambda v: CrossProduct(TriEdges(f, 1, 0, v), TriEdges(f, 2, 0, v)))
                self.tns.v = self.v

                if self.normalized:
                    tmp = MatVecMult(self.faces_by_vertex, self.tns)
                    self.normals = NormalizedNx3(tmp)
                else:
                    test = self.faces_by_vertex.dot(np.ones(self.faces_by_vertex.shape[1]))
                    faces_by_vertex = sp.diags([1. / test], [0]).dot(self.faces_by_vertex).tocsc()
                    normals = MatVecMult(faces_by_vertex, self.tns).reshape((-1, 3))
                    normals = normals / (ch.sum(normals ** 2, axis=1) ** .25).reshape((-1, 1))
                    self.normals = normals

    def compute_r(self):
        return self.normals.r.reshape((-1, 3))

    def compute_dr_wrt(self, wrt):
        if wrt is self.v:
            return self.normals.dr_wrt(wrt)
