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
import pickle

import chumpy as ch
import numpy as np


def smal_horse_prior(prior_pklpath, disable_tail_mouth_ear=True):
    res = pickle.load(open(prior_pklpath, 'rb'), encoding='latin-1')
    if disable_tail_mouth_ear:
        precs = ch.asarray(res['pic'][:81, :81])
        means = ch.asarray(res['mean_pose'][:81])
    else:
        precs = ch.asarray(res['pic'])
        means = ch.asarray(res['mean_pose'])

    def compute(poses):
        return (poses - means).dot(precs)

    return compute


def smal_horse_joint_angle_prior():
    # Indices for the roration angle of  90deg bend at np.pi/2
    # 6, 7, 8,  # LF leg
    # 11, 12, 13,  # RF leg
    # 20, 21, 22,  # LB leg
    # 25, 26, 27  # RB leg

    angle_prior_idxs = np.array([6, 7, 8, 11, 12, 13, 20, 21, 22, 25, 26, 27],
                                dtype=np.int32) - 3  # the pose will be without root

    angle_prior_signs = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], np.float32)

    def compute(pose):
        # assert len(pose) == 105
        return ch.power(ch.exp(pose[angle_prior_idxs] * angle_prior_signs), 2)

    return compute
