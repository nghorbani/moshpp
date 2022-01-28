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
import os.path as osp
from typing import Union

import numpy as np
from human_body_prior.tools.omni_tools import get_support_data_dir


def smplh2smplx(vids: Union[list, str]) -> Union[list, str]:
    support_dir = get_support_data_dir(__file__)

    smplh2smplx = np.load(osp.join(support_dir, 'smplx_fit2_smplh.npz'))['smh2smhf']
    if isinstance(vids, int):
        return int(smplh2smplx[vids])
    return [int(smplh2smplx[vid]) for vid in vids]


def marker_meta_smplh2smplx(smplh_marker_meta: dict) -> dict:
    support_dir = get_support_data_dir(__file__)
    smplh2smplx = np.load(osp.join(support_dir, 'smplx_fit2_smplh.npz'))['smh2smhf']

    assert smplh_marker_meta.get('model_type', 'smplh') == 'smplh', ValueError(
        'unexpected model_type of the given marker layout: {}'.format(smplh_marker_meta['model_type']))

    smplx_marker_meta = {'surface_model_type': 'smplx', 'markersets': []}
    for mrk_set in smplh_marker_meta['markersets']:
        new_mrkset = {}
        for k, v in mrk_set.items():
            if k == 'indices': continue
            new_mrkset[k] = v
        new_indices = {}
        for k, v in mrk_set['indices'].items():

            if v >= len(smplh2smplx):
                new_indices[k] = v
            else:
                new_indices[k] = int(smplh2smplx[v])
        new_mrkset['indices'] = new_indices
        smplx_marker_meta['markersets'].append(new_mrkset)

    # smplx_marker_layout_fname = marker_layout_fname.replace('_smplh.json', '_smplx.json')
    # with open(smplx_marker_layout_fname, 'w') as f:
    #     json.dump(smplx_marker_meta, f, sort_keys=True, indent=4, separators=(',', ': '))
    # print('created {}'.format(smplx_marker_layout_fname))
    # return {'smplx_marker_meta': smplx_marker_meta, 'smplx_marker_layout_fname': smplx_marker_layout_fname}
    return smplx_marker_meta


def marker_meta_smplx2smplh(smplx_marker_meta: dict) -> dict:
    support_dir = get_support_data_dir(__file__)

    smplx2smplh = np.load(osp.join(support_dir, 'smplx_fit2_smplh.npz'))['smhf2smh']

    assert smplx_marker_meta.get('surface_model_type', 'smplx') == 'smplx', ValueError(
        'unexpected model_type of the given marker layout: {}'.format(smplx_marker_meta['model_type']))

    smplh_marker_meta = smplx_marker_meta.copy()

    for l, vid in smplx_marker_meta['marker_vids'].items():
        smplh_marker_meta['marker_vids'][l] = int(smplx2smplh[vid])

    smplh_marker_meta['surface_model_type'] = 'smplh'

    return smplh_marker_meta
