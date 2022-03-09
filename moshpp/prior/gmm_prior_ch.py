# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
# If you use this code in a research publication please cite the following:
#
# @conference{AMASS:ICCV:2019,
#   title = {{AMASS}: Archive of Motion Capture as Surface Shapes},
#   author = {Mahmood, Naureen and Ghorbani, Nima and Troje, Nikolaus F. and Pons-Moll, Gerard and Black, Michael J.},
#   booktitle = {International Conference on Computer Vision},
#   pages = {5442--5451},
#   month = oct,
#   year = {2019},
#   month_numeric = {10}
# }
#
# You can find complementary content at the project website: https://amass.is.tue.mpg.de/
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
# Naureen Mahmood <https://ps.is.tuebingen.mpg.de/person/nmahmood>
# Matthew Loper <https://ps.is.mpg.de/~mloper>
# While at Max-Planck Institute for Intelligent Systems, Tübingen, Germany
#
# 2021.06.18
import os
import pickle
import sys

import chumpy as ch
import numpy as np


class MaxMixtureComplete(ch.Ch):  # only difference is that mahalanobis has the correct 0.5 constant.
    def __init__(self, x, means, precs, weights):
        self.x = x
        self.means = means
        self.precs = precs
        self.weights = weights
        self.loglikelihoods = None
        self.min_component_idx = None

    dterms = 'x', 'means', 'precs', 'weights'

    def on_changed(self, which):
        # setup means, precs and loglikelihood expressions
        if 'means' in which or 'precs' in which or 'weights' in which:
            # This is just the mahalanobis part.
            self.loglikelihoods = [np.sqrt(0.5) * (self.x - m).dot(s) for m, s in zip(self.means, self.precs)]

        if 'x' in which:
            # start = time.time()
            self.min_component_idx = np.argmin([(logl ** 2).sum().r[0] - np.log(w[0])
                                                for logl, w in zip(self.loglikelihoods, self.weights)])
            # print('min took %f' % (time.time() - start))
            # import ipdb; ipdb.set_trace()
            # print([(logl**2).sum().r[0] - w[0] for logl, w in zip(self.loglikelihoods, self.weights)])
            # print('******* mode is %d' % self.min_component_idx)
            # import ipdb; ipdb.set_trace()

    def compute_r(self):
        min_w = self.weights[self.min_component_idx]
        # Add the sqrt(-log(weights)).
        return ch.concatenate((self.loglikelihoods[self.min_component_idx].r, np.sqrt(-np.log(min_w))))

    def compute_dr_wrt(self, wrt):
        # print('shape of wrt: %d' % wrt.shape)

        # Returns 69 x 72, when wrt is 69D => return 70x72 with empty last for when returning 70D
        # Extract the data, rows cols and data, new one with exact same values but with size one more rows)
        import scipy.sparse as sp

        dr = self.loglikelihoods[self.min_component_idx].dr_wrt(wrt)
        if dr is not None:
            Is, Js, Vs = sp.find(dr)
            dr = sp.csc_matrix((Vs, (Is, Js)), shape=(dr.shape[0] + 1, dr.shape[1]))
        return dr


class MaxMixtureCompleteWrapper(object):
    def __init__(self, means, precs, weights, pca_ncomps=6):
        self.means = means
        self.precs = precs  # Already "sqrt"ed
        self.weights = weights
        self.pca_ncomps = pca_ncomps

    def __call__(self, x):
        '''
        idxStart = 0 if self.leftORright == 'left' else self.pca_ncomps
        idxEnddd = idxStart + self.pca_ncomps
        return MaxMixtureComplete(x=x[idxStart:idxEnddd]

        :param x: is the current pose, and it should not include the first 3 elements that represent root joint orientation
        :return:
        '''
        return MaxMixtureComplete(x=x, means=self.means, precs=self.precs, weights=self.weights)


def create_gmm_body_prior(pose_body_prior_fname, exclude_hands=False):
    assert os.path.exists(pose_body_prior_fname), \
        ValueError(f'pose_body_prior_fname does not exist: {pose_body_prior_fname}')

    with open(pose_body_prior_fname, 'rb') as f:
        gmm = pickle.load(f, encoding='latin-1')

    # n_gaussians = gmm.means_.shape[0]

    npose = 63 if exclude_hands else 69

    covars = gmm['covars'][:, :npose, :npose]
    means = gmm['means'][:, :npose]
    weights = gmm['weights']

    precs = ch.asarray([np.linalg.inv(cov) for cov in covars])
    chols = ch.asarray([np.linalg.cholesky(prec) for prec in precs])

    # The constant term:
    sqrdets = np.array([(np.sqrt(np.linalg.det(c))) for c in covars])
    # const = (2*np.pi)**((npose+3)/2.)
    const = (2 * np.pi) ** (npose / 2.)

    weights = weights / (const * (sqrdets / sqrdets.min()))
    # weights = ch.asarray(np.where(weights<1e-15,1e-15,weights))
    weights = ch.asarray(weights)

    return MaxMixtureCompleteWrapper(means=means, precs=chols, weights=weights)


class MaxMixturePriorHands(object):
    def __init__(self, prior_pklpath):
        self.prior_pklpath = prior_pklpath
        self.prior = self.create_prior_from_MPI()

    def create_prior_from_MPI(self):

        if not os.path.exists(self.prior_pklpath):
            raise (ValueError('Hand prior pkl file is not available! %s' % self.prior_pklpath))
        else:
            sys.stderr.write('Hand prior used: %s\n' % self.prior_pklpath)
            with open(self.prior_pklpath) as f:
                gmm = pickle.load(f)

        pca_ncomps = gmm.means_.shape[1]

        precs = ch.asarray([np.linalg.inv(cov) for cov in gmm.covars_])
        chols = ch.asarray([np.linalg.cholesky(prec) for prec in precs])

        # The constant term:
        sqrdets = np.array([(np.sqrt(np.linalg.det(c))) for c in gmm.covars_])
        dim = gmm.means_.shape[1]
        const = (2 * np.pi) ** (dim / 2.)

        weights = ch.asarray(gmm.weights_ / (const * (sqrdets / sqrdets.min())))
        weights = ch.asarray(np.where(weights < 1e-15, 1e-15, weights))

        return MaxMixtureCompleteWrapper(means=gmm.means_, precs=chols, weights=weights, pca_ncomps=pca_ncomps)

    def get_gmm_prior(self):
        return self.prior
