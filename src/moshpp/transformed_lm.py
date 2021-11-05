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
# Naureen Mahmood <https://www.is.mpg.de/person/nmahmood>
# Nima Ghorbani <https://nghorbani.github.io/>
# Naureen Mahmood <https://ps.is.tuebingen.mpg.de/person/nmahmood>
# Matthew Loper <https://ps.is.mpg.de/~mloper>
# While at Max-Planck Institute for Intelligent Systems, Tübingen, Germany
#
# 2021.06.18

import os

import chumpy as ch
import numpy as np
from chumpy import Ch
from human_body_prior.tools.omni_tools import get_support_data_dir
from sklearn.neighbors import NearestNeighbors


class TransformedCoeffs(Ch):
    dterms = 'can_body', 'markers_latent'
    support_base_dir = get_support_data_dir(__file__)
    smplx_eyebals_path = os.path.join(support_base_dir, 'smplx_eyeballs.npz')
    no_eye_ball_vids = list(
        set(np.arange(10474).tolist()).difference(set(np.load(smplx_eyebals_path)['eyeballs'].tolist())))

    def compute_r(self):
        return self._result.r

    def compute_dr_wrt(self, wrt):
        if wrt is self._result:
            return 1

    def on_changed(self, which):

        # print ('Merry Coeffs')

        can_body = self.can_body
        markers_latent = self.markers_latent

        # sknbrs = NearestNeighbors(algorithm='brute', metric='l1', n_neighbors=3).fit(self.can_body.r)
        if len(can_body) == 10475:
            can_body_no_eyeballs = self.can_body.r[TransformedCoeffs.no_eye_ball_vids]
            # sys.stderr.write('Removing eyeballs from SMPLx model.\n')
        else:
            can_body_no_eyeballs = self.can_body.r
        # can_body_no_eyeballs = self.can_body.r if len(can_body) == 6890 else self.can_body.r[TransformedCoeffs.no_eye_ball_vids]
        sknbrs = NearestNeighbors(algorithm='kd_tree', n_neighbors=8).fit(can_body_no_eyeballs)
        _, closest = sknbrs.kneighbors(self.markers_latent.r)
        self.closest = np.vstack(closest)

        # n_mrks = self.markers_latent.shape[0]
        # n_vrts = self.can_body.shape[0]
        # closest_alt = []
        # for mIdx in range(n_mrks):
        #     A = np.repeat(self.markers_latent[mIdx][np.newaxis], repeats=n_vrts, axis=0)
        #     A_distances = np.sqrt(np.sum(np.square(A - can_body),axis=1))
        #     unsorted_closest_idx = np.argpartition(A_distances, 3)[:3]
        #     closest_alt.append(unsorted_closest_idx[np.argsort(A_distances[unsorted_closest_idx])])
        # closest_alt = np.asarray(closest_alt)
        # self.closest = closest_alt

        self.diff = (markers_latent - can_body[self.closest[:, 0]]).reshape((-1, 3))
        self.e1 = can_body[self.closest[:, 1]] - can_body[self.closest[:, 0]]
        self.e2 = can_body[self.closest[:, 2]] - can_body[self.closest[:, 0]]

        self.f1 = nrm(self.e1)

        NN_counter = 3
        while (np.isnan(nrm(ch.cross(self.e1, self.e2)).sum()) and NN_counter < self.closest.shape[0]):
            self.e2 = can_body[self.closest[:, NN_counter]] - can_body[self.closest[:, 0]]
            NN_counter += 1
            print('nearest neighbors are on a line, trying to find next neighbor!!')

        self.closest[:, 2] = self.closest[:, NN_counter - 1]
        self.f2 = nrm(ch.cross(self.e1, self.e2))
        self.f3 = ch.cross(self.f1, self.f2)  # normalizing this is redundant

        project = lambda x, y: (ch.sum(x * y, axis=1)).reshape((-1, 1))
        self.coefs1 = project(self.diff, self.f1)
        self.coefs2 = project(self.diff, self.f2)
        self.coefs3 = project(self.diff, self.f3)

        _result = ch.hstack([self.coefs1, self.coefs2, self.coefs3])
        if not hasattr(self, '_result'):
            self.add_dterm('_result', _result)
        else:
            self._result = _result


def nrm(x):
    return x / ch.sqrt(ch.sum(x ** 2, axis=1)).reshape((-1, 1))


class TransformedLms(Ch):
    dterms = 'transformed_coeffs', 'can_body'

    def compute_r(self):
        return self._result.r

    def compute_dr_wrt(self, wrt):
        if wrt is self._result:
            return 1

    def on_changed(self, which):
        # print ('Merry landmarks')

        _ = self.transformed_coeffs.r  # to get "closest" to pull through
        closest = self.transformed_coeffs.closest
        can_body = self.can_body
        coeffs = self.transformed_coeffs

        e1 = can_body[closest[:, 1]] - can_body[closest[:, 0]]
        e2 = can_body[closest[:, 2]] - can_body[closest[:, 0]]

        f1 = nrm(e1)

        NN_counter = 2
        while (np.isnan(nrm(ch.cross(e1, e2)).sum()) and NN_counter < closest.shape[0]):
            e2 = can_body[closest[:, NN_counter]] - can_body[closest[:, 0]]
            NN_counter += 1
            print('choose 4th nearest neighber instead of 3rd')

        f2 = nrm(ch.cross(e1, e2))
        f3 = ch.cross(f1, f2)  # normalizing this is redundant

        # f1 = self.transformed_coeffs.f1
        # f2 = self.transformed_coeffs.f2
        # f3 = self.transformed_coeffs.f3

        self._result = can_body[closest[:, 0]] + \
                       coeffs[:, 0].reshape((-1, 1)) * f1 + \
                       coeffs[:, 1].reshape((-1, 1)) * f2 + \
                       coeffs[:, 2].reshape((-1, 1)) * f3

        if '_result' not in self.dterms:
            self.add_dterm('_result', self._result)
