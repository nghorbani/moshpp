import os
import pickle

import chumpy as ch
import numpy as np

from moshpp.prior.gmm_prior_ch import MaxMixtureComplete


class MaxMixtureCompleteWrapper(object):
    def __init__(self, means, precs, weights, pca_ncomps=6):
        self.means = means
        self.precs = precs  # Already "sqrt"ed
        self.weights = weights
        self.pca_ncomps = pca_ncomps
        # self.joint_ids = joint_ids

    def __call__(self, x):
        '''
        idxStart = 0 if self.leftORright == 'left' else self.pca_ncomps
        idxEnddd = idxStart + self.pca_ncomps
        return MaxMixtureComplete(x=x[idxStart:idxEnddd]

        :param x: is the current pose, and it should not include the first 3 elements that represent root joint orientation
        :return:
        '''
        # if self.joint_ids is not None:
        #     x = x[self.joint_ids]
        return MaxMixtureComplete(x=x, means=self.means, precs=self.precs, weights=self.weights)


def smal_dog_prior(prior_pklpath):
    res = pickle.load(open(prior_pklpath, 'rb'), encoding='latin-1')
    # if disable_tail_mouth_ear:
    #     precs = ch.asarray(res['pic'][:81,:81])
    #     means = ch.asarray(res['mean_pose'][:81])
    # else:
    precs = ch.asarray(res['cov'])
    means = ch.asarray(res['means'])

    def compute(poses):
        return (poses - means).dot(precs)

    return compute


class MaxMixtureDog(object):

    def __init__(self, prior_pklpath):
        self.prior_pklpath = prior_pklpath
        self.weights = None

    def get_gmm_prior(self):
        if self.prior_pklpath is None:
            return None
        joint_ids = [1, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                     30, 31, 32, 33, 34]
        joint_ids = np.arange(0, 105).reshape([-1, 3])[joint_ids].reshape(-1)

        if not os.path.exists(self.prior_pklpath):
            raise (ValueError(f"Body prior pkl file is not available! {self.prior_pklpath}"))
        else:
            with open(self.prior_pklpath, 'rb') as f:
                gmm = pickle.load(f, encoding='latin-1')

        # n_gaussians = gmm.means_.shape[0]
        npose = len(joint_ids)
        # print(gmm.keys())
        covars = gmm['gmm_covs'][:, :, joint_ids][:, joint_ids]  # apears to be a bug in py2.7 ppf numpy
        means = gmm['gmm_means'][:, joint_ids]
        weights = gmm['gmm_weights'][:]

        precs = ch.asarray([np.linalg.inv(cov) for cov in covars])
        chols = ch.asarray([np.linalg.cholesky(prec) for prec in precs])

        # The constant term:
        sqrdets = np.array([(np.sqrt(np.linalg.det(c))) for c in covars])
        assert np.any(sqrdets == 0.0), ZeroDivisionError(
            f'Encountered zeros in the determinant of the covariance matrix:  {sqrdets}')
        # const = (2*np.pi)**((npose+3)/2.)
        const = (2 * np.pi) ** (npose / 2.)

        weights = weights / (const * (sqrdets / sqrdets.min()))
        # weights = ch.asarray(np.where(weights<1e-15,1e-15,weights))
        weights = ch.asarray(weights)

        return MaxMixtureCompleteWrapper(means=means, precs=chols, weights=weights)

# def smal_dog_joint_angle_prior():
#     # Indices for the roration angle of  90deg bend at np.pi/2
#     # 6, 7, 8,  # LF leg
#     # 11, 12, 13,  # RF leg
#     # 20, 21, 22,  # LB leg
#     # 25, 26, 27  # RB leg
#
#     angle_prior_idxs = np.array([6, 7, 8, 11, 12, 13, 20, 21, 22, 25, 26, 27], dtype=np.int32) - 3 #the pose will be without root
#
#     angle_prior_signs = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], np.float32)
#
#     def compute(pose):
#         #assert len(pose) == 105
#         return ch.power(ch.exp(pose[angle_prior_idxs] *  angle_prior_signs),2)
#
#     return compute
