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
import copy
import json
from os import path as osp

import numpy as np
from human_body_prior.tools.omni_tools import rm_spaces
from omegaconf import OmegaConf


def universal_mosh_jobs_filter(total_jobs, only_stagei=False, determine_shape_for_each_seq=False):
    from moshpp.mosh_head import MoSh

    filtered_jobs = []
    exclude_keys = []
    for cur_job in total_jobs:
        mocap_key = '_'.join(cur_job['mocap.fname'].split('/')[-3:-1])
        mosh_cfg = MoSh.prepare_cfg(**copy.deepcopy(cur_job))

        if mocap_key in exclude_keys: continue
        if osp.exists(mosh_cfg.dirs.stageii_fname): continue  # mosh is complete

        if not osp.exists(mosh_cfg.dirs.stagei_fname) \
                and not determine_shape_for_each_seq: exclude_keys.append(mocap_key)
        if only_stagei and osp.exists(mosh_cfg.dirs.stagei_fname): continue
        filtered_jobs.append(cur_job)
    return filtered_jobs


def turn_fullpose_into_parts(fullpose, surface_model_type):
    res = {'root_orient': fullpose[:, :3]}
    if 'smpl' in surface_model_type:
        res['pose_body'] = fullpose[:, 3:66]
    elif np.any([text in surface_model_type for text in ['animal', 'object']]):
        res['pose_body'] = fullpose[:, 3:]

    if 'smplh' in surface_model_type:
        res['pose_hand'] = fullpose[:, 66:]
    elif 'smplx' in surface_model_type:
        res['pose_hand'] = fullpose[:, 75:]
        res['pose_jaw'] = fullpose[:, 66:69]
        res['pose_eye'] = fullpose[:, 69:75]
    elif 'mano' in surface_model_type:
        res['pose_hand'] = fullpose[:, 3:]
    return res


def setup_mosh_omegaconf_resolvers():
    """
    ds_name, subject name and mocap basename are automatically extracted from the mocap path.
    i.e. ...\ds_name\subject_name\mocap_basename.c3d

    The subject gender is assumed to be give in a json file inside the subject folder.
    The file should be settings.json and the content as an example should be {'gender': female}

    """
    def resolve_mosh_subject_gender(mocap_fname, fall_back_gender):
        gender_fname = osp.join(osp.dirname(mocap_fname), 'settings.json')

        if osp.exists(gender_fname):
            gender = json.load(open(gender_fname))['gender']
        else:
            if fall_back_gender == 'error':
                raise FileNotFoundError(f'gender file not available at {gender_fname}')
            else:
                gender = fall_back_gender

        return gender

    if not OmegaConf.has_resolver('resolve_mocap_subject'):
        OmegaConf.register_new_resolver('resolve_mocap_subject',
                                        lambda mocap_fname: rm_spaces(mocap_fname.split('/')[-2]))

    if not OmegaConf.has_resolver('resolve_mocap_basename'):
        OmegaConf.register_new_resolver('resolve_mocap_basename',
                                        lambda mocap_fname: rm_spaces(
                                            '.'.join(mocap_fname.split('/')[-1].split('.')[:-1])))

    if not OmegaConf.has_resolver('resolve_mocap_ds_name'):
        OmegaConf.register_new_resolver('resolve_mocap_ds_name',
                                        lambda mocap_fname: rm_spaces(mocap_fname.split('/')[-3]))

    if not OmegaConf.has_resolver('resolve_gender'):
        OmegaConf.register_new_resolver('resolve_gender',
                                        lambda mocap_fname, fall_back_gender='error': resolve_mosh_subject_gender(
                                            mocap_fname,
                                            fall_back_gender))
