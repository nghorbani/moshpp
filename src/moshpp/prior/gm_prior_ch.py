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
# While at Max-Planck Institute for Intelligent Systems, Tübingen, Germany
#
# 2021.06.18
class Mahalanobis(object):

    def __init__(self, mean, prec, prefix):
        self.mean = mean
        self.prec = prec
        self.prefix = prefix

    def __call__(self, pose):
        return (pose[self.prefix:] - self.mean).reshape(1, -1).dot(self.prec)


class Prior(object):

    def __init__(self, model_pklpath, prefix=3):
        from psbody.smpl import load_model

        self.prefix = prefix
        model = load_model(model_pklpath)

        self.pose_subjects = model.pose_subjects
        all_samples = [p[prefix:] for qsub in self.pose_subjects
                       for name, p in zip(qsub['pose_fnames'], qsub['pose_parms'])]
        self.priors = {'Generic': self.create_prior_from_samples(all_samples)}

    def create_prior_from_samples(self, samples):
        from sklearn.covariance import GraphLassoCV
        from numpy import asarray, linalg
        from chumpy import Ch
        model = GraphLassoCV()
        model.fit(asarray(samples))
        return Mahalanobis(asarray(samples).mean(axis=0),
                           Ch(linalg.cholesky(model.precision_)),
                           self.prefix)

    def __getitem__(self, pid):
        if pid not in self.priors:
            # Gather in the training data of the models, the ones that contain the prior string in the filename
            samples = [p[self.prefix:] for qsub in self.pose_subjects
                       for name, p in zip(qsub['pose_fnames'],
                                          qsub['pose_parms'])
                       if pid.lower() in name.lower()]
            # if there are more than 3, use them to create a new prior
            self.priors[pid] = self.priors['Generic'] if len(samples) < 3 else self.create_prior_from_samples(samples)

        return self.priors[pid]
