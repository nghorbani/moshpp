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

"""
SMPL fitting wit fast derivatives
---------------------------------
"""
import os.path  as osp
import pickle as pickle

import chumpy as ch
import numpy as np
import scipy.sparse as sp
from psbody.mesh import Mesh
from psbody.smpl.fast_derivatives.smplcpp_chumpy import lbs_derivatives_wrt_pose, lbs_derivatives_wrt_shape
from psbody.smpl.verts import verts_decorated
from loguru import logger

def load_surface_model(surface_model_fname,
                       pose_hand_prior_fname=None,
                       use_hands_mean=False,
                       dof_per_hand=12,
                       v_template_fname=None,
                       body_parms={}):
    assert surface_model_fname.endswith('.pkl'), ValueError('surface_model_fname could only be a pkl file.')

    dd = pickle.load(open(surface_model_fname, 'rb'), encoding='latin-1')

    njoint_parms = dd['posedirs'].shape[2] // 3
    model_type = {69: 'smpl', 153: 'smplh', 162: 'smplx', 45: 'mano', 105: 'animal_horse'}[njoint_parms]

    # if pose_hand_prior_fname is not None:
    #     assert model_type in ['mano', 'smplx', 'smplh']

    if v_template_fname is not None:
        assert osp.exists(v_template_fname), FileExistsError(v_template_fname)
        dd['v_template'] = Mesh(filename=v_template_fname).v
        logger.info(f'Using v_template_fname: {v_template_fname}')

    if model_type in ['smplx', 'smplh']:
        pose_body_dof = njoint_parms - 90 + 3

        assert pose_hand_prior_fname is not None and pose_hand_prior_fname.endswith('.npz')

        mano_prior_parms = np.load(pose_hand_prior_fname)

        hands_componentsl = mano_prior_parms['componentsl']  # (45, 45)
        hands_meanl = mano_prior_parms['hands_meanl'] if use_hands_mean else np.zeros(
            hands_componentsl.shape[1])  # (45,)
        # hands_coeffsl = mano_prior_parms['hands_coeffsl'][:,:dof_per_hand]  # (659, 45)  ---> (659, 6)

        hands_componentsr = mano_prior_parms['componentsr']  # (45, 45)
        hands_meanr = mano_prior_parms['hands_meanr'] if use_hands_mean else np.zeros(
            hands_componentsr.shape[1])  # (45,)
        # hands_coeffsr = mano_prior_parms['hands_coeffsr'][:,:dof_per_hand]  # (895, 45)  ---> (895, 6)

        selected_components = np.vstack(
            (np.hstack((hands_componentsl[:dof_per_hand], np.zeros_like(hands_componentsl[:dof_per_hand]))),
             np.hstack((np.zeros_like(hands_componentsr[:dof_per_hand]), hands_componentsr[:dof_per_hand]))))
        hands_mean = np.concatenate((hands_meanl, hands_meanr))

        dd['pose'] = ch.zeros(pose_body_dof + selected_components.shape[0])

        dd['pose_body_dof'] = pose_body_dof
        dd['pose_hand_dof'] = dof_per_hand * 2
        dd['selected_components'] = selected_components
        dd['hands_mean'] = hands_mean

    elif model_type in ['mano']:
        pose_body_dof = 3

        hands_components = dd['hands_components']
        hands_mean = np.zeros(hands_components.shape[1]) if use_hands_mean else dd['hands_mean']
        # hands_coeffs = dd['hands_coeffs'][:, :dof_per_hand]

        selected_components = np.vstack((hands_components[:dof_per_hand]))
        dd['pose_body_dof'] = pose_body_dof

        dd['pose_hand_dof'] = dof_per_hand
        dd['selected_components'] = selected_components
        dd['hands_mean'] = hands_mean

        dd['pose'] = ch.zeros(pose_body_dof + selected_components.shape[0])

    else:
        dd['pose'] = ch.zeros(njoint_parms + 3)

    Jreg = dd['J_regressor']
    if not sp.issparse(Jreg):
        dd['J_regressor'] = (sp.csc_matrix((Jreg.data, (Jreg.row, Jreg.col)), shape=Jreg.shape))

    dd['model_type'] = model_type
    dd['trans'] = ch.zeros(3)
    if 'betas' not in body_parms:
        dd['betas'] = np.zeros(dd['shapedirs'].shape[-1])
    else:
        dd['betas'] = body_parms['betas']

    for s in ['trans', 'pose', 'betas', 'v_template', 'weights', 'posedirs', 'shapedirs']:
        if (s in dd) and not hasattr(dd[s], 'dterms'):
            dd[s] = ch.array(dd[s])

    dd.update({
        'xp': ch,
        'want_Jtr': True,
    })

    class Struct:
        pass

    model = Struct()
    for k, v in dd.items():
        model.__setattr__(k, v)

    model_with_fast_derivatives = SmplModelLBS(pose=dd['pose'],
                                               trans=dd['trans'],
                                               betas=dd['betas'],
                                               temp_model=model)  # Smpl model based on linear blend skinning.

    return model_with_fast_derivatives


class SmplModelLBS(ch.Ch):
    dterms = ['trans', 'pose', 'betas']

    # low_res is used to obtain a lower-res temp_model; it is a dict with keys 'vids' and 'faces'
    # (vertex ids to retain, faces for the lower-res temp_model)
    def __init__(self, trans, pose, betas, temp_model):

        assert (temp_model.bs_style == 'lbs')

        self.model_type = temp_model.model_type

        if self.model_type in ['smplh', 'smplx', 'mano']:
            self.selected_components = temp_model.selected_components
            self.hands_mean = temp_model.hands_mean  # Hands Mean
            self.pose_hand_dof = temp_model.pose_hand_dof  # Number of components
            self.pose_body_dof = temp_model.pose_body_dof  # body pose degree of freedom
            # Todo: shouldn't this be self.pose?
            full_hand_pose = pose[self.pose_body_dof:(self.pose_body_dof + self.pose_hand_dof)].dot(
                self.selected_components)

            full_body_pose = ch.concatenate((pose[:self.pose_body_dof], self.hands_mean + full_hand_pose))
        else:
            full_body_pose = self.pose

        self._inner_model = verts_decorated(trans=self.trans,
                                            pose=full_body_pose,
                                            v_template=temp_model.v_template,
                                            J=temp_model.J_regressor,
                                            weights=temp_model.weights,
                                            kintree_table=temp_model.kintree_table,
                                            bs_style=temp_model.bs_style,
                                            f=temp_model.f,
                                            bs_type=temp_model.bs_type,
                                            posedirs=temp_model.posedirs,
                                            betas=self.betas,
                                            shapedirs=temp_model.shapedirs[:, :, :len(self.betas)])

        self.v_shaped = self._inner_model.v_shaped
        self.bs_style = self._inner_model.bs_style
        self.posedirs = self._inner_model.posedirs
        self.bs_type = self._inner_model.bs_type
        self.shapedirs = self._inner_model.shapedirs
        # self.A = self._inner_model.A
        # self.A_global = self._inner_model.A_global
        self.J_transformed = self._inner_model.Jtr
        self.J = self._inner_model.J
        self.Jtr = self._inner_model.Jtr
        self.f = self._inner_model.f
        self.weights = self._inner_model.weights
        self.v_posed = self._inner_model.v_posed
        # self.A_weighted = self._inner_model.A_weighted
        self.kintree_table = self._inner_model.kintree_table
        self.v_template = temp_model.v_template
        self.J_regressor = temp_model.J_regressor
        self.fullpose = full_body_pose

        # required for betas derivatives
        self._inner_model.J_regressor = temp_model.J_regressor
        if hasattr(temp_model, 'priors'): self.priors = temp_model.priors

    def compute_r(self):
        return self._inner_model.r

    def compute_dr_wrt(self, wrt):
        if wrt is self.pose:
            if self.model_type in ['smplh', 'smplx', 'mano']:
                import numpy as np
                row1 = np.hstack([np.eye(self.pose_body_dof), np.zeros(
                    (self.pose_body_dof,
                     self.pose_hand_dof))])  # for SMPL+HF ---> np.hstack([np.eye(75),np.zeros((75,12))])
                row2 = np.hstack([np.zeros((self.selected_components.shape[1], self.pose_body_dof)), np.transpose(
                    self.selected_components)])  # for SMPL+HF ---> np.hstack([np.zeros((90,75)),np.transpose(self.selected_components)])
                m = np.vstack((row1, row2))
                g = np.matmul(lbs_derivatives_wrt_pose(self._inner_model), m)
                return g
            else:
                return lbs_derivatives_wrt_pose(self._inner_model)

        if wrt is self.betas:
            return lbs_derivatives_wrt_shape(self._inner_model)

        return self._inner_model.dr_wrt(wrt)


if __name__ == '__main__':
    pose_hand_prior_fname = '/ps/scratch/common/moshpp/moshpp3/smplx/pose_hand_prior.npz'

    surface_model_fname = '/ps/scratch/common/moshpp/smplx/locked_head/model_6_merged_exp_hands_fixed_eyes/neutral/model.pkl'
    # surface_model_fname = '/ps/scratch/common/moshpp/smplh/locked_head/neutral/model.pkl'
    # surface_model_fname = '/ps/scratch/common/moshpp/smpl/locked_head/neutral/model.pkl'
    # surface_model_fname = '/ps/scratch/common/moshpp/mano/MANO_RIGHT.pkl'
    bm = load_surface_model(surface_model_fname, pose_hand_prior_fname=pose_hand_prior_fname)
    dr = bm.compute_dr_wrt(bm.betas)
    print(bm.r.shape, bm.pose.shape, bm.fullpose.shape)
