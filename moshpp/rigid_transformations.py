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
import cv2
import numpy as np
import scipy


def rigid_landmark_transform(a, b):
    """
    Args:
        a: a 3xN array of vertex locations
        b: a 3xN array of vertex locations

    Returns: (R,T) such that R.dot(a)+T ~= b
    Based on Arun et al, "Least-squares fitting of two 3-D point sets," 1987.
    See also Eggert et al, "Estimating 3-D rigid body transformations: a
    comparison of four major algorithms," 1997.
    """
    assert (a.shape[0] == 3)
    assert (b.shape[0] == 3)
    b = np.where(np.isnan(b), a, b)
    a_mean = np.mean(a, axis=1).reshape((-1, 1))
    b_mean = np.mean(b, axis=1).reshape((-1, 1))
    a_centered = a - a_mean
    b_centered = b - b_mean

    c = a_centered.dot(b_centered.T)
    u, s, v = np.linalg.svd(c, full_matrices=False)
    v = v.T
    R = v.dot(u.T)

    if scipy.linalg.det(R) < 0:
        v[:, 2] = -v[:, 2]
        R = v.dot(u.T)

    T = (b_mean - R.dot(a_mean)).reshape((-1, 1))

    return (R, T)


def perform_rigid_adjustment(poses, trans, opt_models, markers_obs, markers_sim):
    for sv_idx, _ in enumerate(opt_models):

        obs_mrk = markers_obs[sv_idx]
        sim_mrk = markers_sim[sv_idx]
        if isinstance(sim_mrk, np.ndarray):
            R, T = rigid_landmark_transform(sim_mrk.T, obs_mrk.T)
        else:
            R, T = rigid_landmark_transform(sim_mrk.r.T, obs_mrk.T)

        poses[sv_idx][:3] = cv2.Rodrigues(R)[0].ravel()
        trans[sv_idx][:] = T.ravel()
