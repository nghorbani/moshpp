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
#
# If you use this code in a research publication please consider citing the following:
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
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
# Naureen Mahmood <https://ps.is.tuebingen.mpg.de/person/nmahmood>
# Matthew Loper <https://ps.is.mpg.de/~mloper>
# While at Max-Planck Institute for Intelligent Systems, Tübingen, Germany
#
# 2018.12.19

import pickle
from pathlib import Path
from typing import List, Dict, Union

import chumpy as ch
import numpy as np
from human_body_prior.tools.omni_tools import flatten_list
from loguru import logger
from omegaconf import DictConfig
from psbody.mesh import Mesh
from sklearn.neighbors import NearestNeighbors

from moshpp.marker_layout.edit_tools import marker_layout_load
from moshpp.marker_layout.labels_map import general_labels_map
from moshpp.models.bodymodel_loader import load_moshpp_models
from moshpp.rigid_transformations import perform_rigid_adjustment
from moshpp.scan2mesh.mesh_distance_main import PtsToMesh
from moshpp.tools.mocap_interface import MocapSession
from moshpp.transformed_lm import TransformedCoeffs, TransformedLms
from moshpp.visualization import visualize_shape_estimate, visualize_pose_estimate


def prepare_mosh_markers_latent(can_model, marker_meta):
    can_v = can_model.r
    can_mesh = Mesh(v=can_v, f=can_model.f)
    can_vn = can_mesh.estimate_vertex_normals()

    m2b_dist = np.ones(len(marker_meta['marker_vids'])) * 0.0095
    for mask_type, marker_mask in marker_meta['marker_type_mask'].items():
        m2b_dist[marker_mask] = marker_meta['m2b_distance'][mask_type]

    vids = list(marker_meta['marker_vids'].values())
    markers_latent = ch.array(can_v[vids] + can_vn[vids] * m2b_dist[:, None])

    surface_distance = PtsToMesh(
        sample_verts=markers_latent,
        reference_verts=can_model,
        reference_faces=can_mesh.f,
        reference_template_or_sampler=can_mesh,
        rho=lambda x: x,
        normalize=False,  # want in meters, so don't normalize
        signed=True)

    desired_distances = ch.array(np.array(m2b_dist))

    return markers_latent, surface_distance - desired_distances


def mosh_stagei(stagei_frames: List[Dict[str, np.ndarray]], cfg: DictConfig,
                betas_fname: Union[Path, str] = None,
                v_template_fname: Union[Path, str] = None) -> dict:
    """
    This is supposed to be used for estimation of subject shape from a list of a single subject performed mocap sessions
    it can also be used for fine tuning of the can markers.
    When using hand it is assumed that the finger markers dont move away that far from thir initial guess so can markers upto body markers are updated

    :return:
    """
    if betas_fname is not None:
        assert betas_fname.endswith('.npz')
        betas = np.load(betas_fname)['betas']
    else:
        betas = None

    # Todo: check for it if given vtempalte the values for betas should be zeros
    num_train_markers = 46  # constant

    if cfg.surface_model.type == 'smplx' and cfg.moshpp.optimize_betas and (
            cfg.mocap.exclude_marker_types is None or 'face' not in cfg.mocap.exclude_marker_types):
        logger.info(
            'For optimizing face markers in smplx chumpy implementation you should set optimize_betas to False.\n'
            'Otherwise face markers must be excluded and optimize_face must be false. '
            'Chumpy implementation does not allow shared betas and '
            'separate facial expressions for first stage. You can run stagei twice as a fix. \n'
            'In the first run you get the shape parameters and in the second run you get face marker placement.'
            'This wont be accurate.')
        logger.info('Adding face to mocap.exclude_markers')
        if cfg.mocap.exclude_marker_types is None:
            cfg.mocap.exclude_marker_types = ['face']
        else:
            cfg.mocap.exclude_marker_types.append(['face'])
        logger.info('Setting moshpp.optimize_face to False')
        cfg.moshpp.optimize_face = False

    logger.info(f'using marker_layout_fname: {cfg.dirs.marker_layout_fname}')
    marker_meta = marker_layout_load(cfg.dirs.marker_layout_fname, include_nan=True,
                                     exclude_markers=cfg.mocap.exclude_markers,
                                     exclude_marker_types=cfg.mocap.exclude_marker_types,
                                     only_markers=cfg.mocap.only_markers, labels_map=general_labels_map)

    optimize_betas = cfg.moshpp.optimize_betas

    # 2. Loading SMPL models.
    # Canonical model is for canonical(a.k.a can space). the beta params of the can_model are ultimately used
    # Optimization models are associated with each frame

    can_model, opt_models = load_moshpp_models(surface_model_fname=cfg.surface_model.fname,
                                               surface_model_type=cfg.surface_model.type,
                                               num_beta_shared_models=cfg.moshpp.stagei_frame_picker.num_frames,
                                               pose_hand_prior_fname=cfg.moshpp.pose_hand_prior_fname,
                                               pose_body_prior_fname=cfg.moshpp.pose_body_prior_fname,
                                               use_hands_mean=cfg.surface_model.use_hands_mean,
                                               dof_per_hand=cfg.surface_model.dof_per_hand,
                                               v_template_fname=v_template_fname)

    assert marker_meta['surface_model_type'] == can_model.model_type == cfg.surface_model.type

    if hasattr(can_model, 'betas') and (betas is not None):
        # if cfg.surface_model.type == 'smplx':
        #     can_model.body_shape[:cfg.surface_model.num_betas] = prev_shape_est['betas'][:cfg.surface_model.num_betas].copy()
        # else:
        can_model.betas[:cfg.surface_model.num_betas] = betas[:cfg.surface_model.num_betas].copy()
        logger.debug(f'Will use previously computed betas and optimize_betas = {optimize_betas}')

    # 4. Get initial guess for the can marker and set them up
    if cfg.mocap.exclude_markers: logger.debug(f'exclude_markers: {cfg.mocap.exclude_markers}')

    markers_latent, distance_to_surface_obj = prepare_mosh_markers_latent(can_model=can_model, marker_meta=marker_meta)
    latent_labels = list(marker_meta['marker_vids'].keys())

    logger.debug(f'Estimating for #latent markers: {len(markers_latent)}')

    tc = TransformedCoeffs(can_body=can_model, markers_latent=markers_latent)
    tc.markers_latent = markers_latent
    tc.can_body = can_model

    # list of chained objects to update estimated markers w.r.t transformed model verts in 12 sample (posed) frames
    markers_sim_all = [TransformedLms(transformed_coeffs=tc, can_body=model) for model in opt_models]

    # Init Markers
    tc2 = TransformedCoeffs(can_body=can_model.r, markers_latent=markers_latent.r)
    init_markers_latent = TransformedLms(transformed_coeffs=tc2, can_body=can_model)
    # todo: couldn't you simply call init_markers_latent = markers_latent.r?

    lm_diffs = []
    markers_sim = []
    markers_obs = []
    labels_obs = []

    for fIdx, obs_frame_data in enumerate(stagei_frames):
        # cur_frame = {}
        obs_labels = [k for k,v in obs_frame_data.items() if not np.any(np.isnan(v))]

        common_labels = list(set(latent_labels).intersection(set(obs_labels)))
        obf = np.vstack([obs_frame_data[k] for k in common_labels])
        lm_ids = [latent_labels.index(k) for k in common_labels]
        smf = markers_sim_all[fIdx][lm_ids]

        markers_obs.append(obf)
        markers_sim.append(smf)
        labels_obs.append(common_labels)
        lm_diffs.append(obf-smf)

        # for obs_label, obs_marker in obs_frame_data.items():
        #     if np.any(np.isnan(obs_marker)): continue
        #     if obs_label not in latent_labels: continue
        #     assert obs_label not in cur_frame, ValueError(
        #         f'Repeated label ({obs_label}) in {stagei_frames[fIdx]}')
        #     latent_id = latent_labels.index(obs_label)
        #     cur_frame[obs_label] = (obs_marker, markers_sim_all[fIdx][latent_id])
        # markers_obs.append([d[0] for d in cur_frame.values()])
        # markers_sim.append([d[1] for d in cur_frame.values()])
        # labels_obs.append([d for d in cur_frame.keys()])
        # lm_diffs.append(ch.vstack([d[0] - d[1] for d in cur_frame.values()]))

    data_obj = ch.vstack(lm_diffs)

    logger.debug('Number of available markers in each stagei selected frames: {}'.format(
        ', '.join([f'(F{fIdx:02d}, {len(frame)})' for fIdx, frame in enumerate(markers_obs)])
    ))

    unavailable_latent_labels = list(set(latent_labels).difference(set(flatten_list(labels_obs))))
    if len(unavailable_latent_labels) != 0:
        logger.debug(
            f'Some labels in the provided marker layout did not exist in the observed frames: {unavailable_latent_labels}')

    # Rigidly adjust poses/trans to fit bodies to landmarks
    logger.debug('Rigidly aligning the body to the markers')
    poses = [model.pose for model in opt_models]
    trans = [model.trans for model in opt_models]
    perform_rigid_adjustment(poses, trans, opt_models, markers_obs, markers_sim)

    if cfg.opt_settings.extra_initial_rigid_adjustment:
        ch.minimize(fun=data_obj, x0=[p[:3] for p in poses] + trans, method='dogleg',
                    options={'e_3': .001, 'delta_0': 5e-1, 'disp': 0, 'maxiter': cfg.opt_settings.maxiter})

    # 10. Setup visualization methods
    if cfg.moshpp.verbosity > 1:
        on_step = visualize_shape_estimate(opt_models=opt_models,
                                           can_model=can_model,
                                           markers_sim=markers_sim,
                                           markers_obs=markers_obs,
                                           markers_latent=markers_latent,
                                           init_markers_latent=init_markers_latent,
                                           marker_meta=marker_meta)
    else:
        on_step = None

    # # Set up objective
    stagei_wts = cfg.opt_settings.weights
    logger.debug('MoSh stagei weights before annealing:\n' +
                 '\n'.join([f'{k}: {wt}' for k, wt in stagei_wts.items() if k.startswith('stagei_wt')]))

    head_mrk_corr = None
    if cfg.moshpp.head_marker_corr_fname is not None:  # and 'head' in can_meta['mrk_ids']:
        logger.debug('head_marker_corr_fname is provided and is being loaded.')
        head_meta = np.load(cfg.moshpp.head_marker_corr_fname)
        head_marker_availability = {m: m in marker_meta['marker_vids'] for m in head_meta['mrk_labels']}
        if np.all(list(head_marker_availability.values())):
            head_mrk_ids = [latent_labels.index(m) for m in head_meta['mrk_labels']]

            head_mrk_corr = ch.asarray(head_meta['corr'])
            logger.debug(
                f'Taking into account the correlation of the head markers from head_marker_corr_fname = {cfg.moshpp.head_marker_corr_fname}')
        else:
            logger.debug('Not all of the head markers are available to take cov into account: {}'.format(' -- '.join(
                [f'({k}, {v})' for k, v in head_marker_availability.items()]
            )))

    logger.debug(f'Beginning mosh stagei with opt_settings.weights_type: {cfg.opt_settings.weights_type}')

    # Setup Variables
    v_betas = []
    if optimize_betas and hasattr(can_model, 'betas'):
        v_betas = [can_model.betas[:cfg.surface_model.num_betas]]

    v_face_exp = []
    pose_ids = list(range(can_model.pose.size))
    pose_body_ids = []
    pose_finger_ids = []
    pose_face_ids = []
    pose_root_ids = pose_ids[:3]
    if cfg.surface_model.type == 'smpl':
        pose_body_ids = pose_ids[3:]
    elif cfg.surface_model.type == 'smplh':
        pose_body_ids = pose_ids[3:66]
        if cfg.moshpp.optimize_fingers:  # dont chop chumpy variables two times
            pose_finger_ids = pose_ids[66:]
    elif cfg.surface_model.type == 'smplx':  # orient:3, body:63, jaw:3, eyel:3, eyer:3, handl, handr
        pose_body_ids = pose_ids[3:66]
        if cfg.moshpp.optimize_face:
            # if optimize_betas:
            #     raise NotImplementedError(
            #         'MoSh requires in the shape stage a single (shared) beta across frames and different per-frame facial expressions.'
            #         ' The current chumpy implementation of smplx does not support this. So if you want to optimize the face you need to provide the shape, or provide a v_template.')

            pose_face_ids = pose_ids[66:69]
            # only jaw, the 69:75 represent eye balls and it might be a bit difficult to capture gaze from mocap,

            v_face_exp = [model.betas[100:100 + cfg.surface_model.num_expressions] for model in opt_models]
        if cfg.moshpp.optimize_fingers:
            pose_finger_ids = pose_ids[75:]
    elif cfg.surface_model.type == 'mano':
        pose_finger_ids = pose_ids[3:]

    detailed_step = False

    for tidx, wt_anneal_factor in enumerate(stagei_wts.stagei_wt_annealing):
        if tidx > len(stagei_wts.stagei_wt_annealing) - 3: detailed_step = True

        opt_objs = {}

        if len(pose_body_ids): wt_poseB = stagei_wts.stagei_wt_poseB * wt_anneal_factor
        if len(pose_finger_ids): wt_poseH = stagei_wts.stagei_wt_poseH * wt_anneal_factor
        if len(pose_face_ids):
            wt_expr = stagei_wts.stagei_wt_expr * wt_anneal_factor
            wt_poseF = stagei_wts.stagei_wt_poseF * wt_anneal_factor
        if len(v_betas): wt_beta = stagei_wts.stagei_wt_betas * wt_anneal_factor

        wt_data = (stagei_wts.stagei_wt_data / wt_anneal_factor) * (num_train_markers / len(latent_labels))

        wt_init = {k: stagei_wts.get(f'stagei_wt_init_{k}', stagei_wts.stagei_wt_init) * wt_anneal_factor for k in
                   marker_meta['marker_type_mask'].keys()}
        wt_surf = {k: stagei_wts.get(f'stagei_wt_surf_{k}', stagei_wts.stagei_wt_surf) for k in
                   marker_meta['marker_type_mask'].keys()}

        wt_messages = f'Step {tidx + 1}/{len(stagei_wts.stagei_wt_annealing)} :' \
                      f' Opt. wt_anneal_factor = {wt_anneal_factor:.2f}, ' \
                      f'wt_data = {wt_anneal_factor:.2f}, wt_poseB = {wt_data:.2f}'
        if cfg.moshpp.optimize_fingers:
            wt_messages += f', wt_poseH = {wt_poseH:.2f}'
        if cfg.moshpp.optimize_face:
            wt_messages += f', wt_expr = {wt_expr:.2f}'

        logger.debug(wt_messages)
        logger.debug(
            f'stagei_wt_init for different marker types {", ".join([f"{k} = {v:.02f}" for k, v in wt_init.items()])}: ')
        logger.debug(
            f'stagei_wt_surf for different marker types {", ".join([f"{k} = {v:.02f}" for k, v in wt_surf.items()])}')

        # opt_objs.update({'data_%s' % k: data_obj[k] * wt_data[k] for k in can_meta['mrk_ids']})
        opt_objs['data'] = data_obj * wt_data

        if len(pose_body_ids):
            opt_objs['poseB'] = ch.concatenate(
                [model.priors['pose'](model.pose[pose_body_ids]) for model in opt_models]) * wt_poseB

        init_loss = (markers_latent - init_markers_latent)
        # opt_objs['init'] = wt_init * init_loss
        if head_mrk_corr is not None:
            opt_objs.update(
                {'init_%s' % k: init_loss[list(
                    set(np.arange(len(latent_labels))[marker_type_mask]).difference(head_mrk_ids))] * wt_init[k] for
                 k, marker_type_mask in
                 marker_meta['marker_type_mask'].items() if k != 'head'})
            opt_objs['init_head_corr'] = head_mrk_corr.dot(init_loss[head_mrk_ids]) * wt_init.get('body',
                                                                                                  stagei_wts.stagei_wt_init * wt_anneal_factor)

        else:
            opt_objs.update({f'init_{k}': init_loss[marker_type_mask] * wt_init[k] for k, marker_type_mask in
                             marker_meta['marker_type_mask'].items()})

        if len(v_betas): opt_objs['beta'] = can_model.priors['betas'].ravel() * wt_beta
        # opt_objs['surf'] = surf_loss * wt_surf
        opt_objs.update(
            {f'surf_{k}': distance_to_surface_obj[marker_type_mask] * wt_surf[k] for k, marker_type_mask in
             marker_meta['marker_type_mask'].items()})
        # opt_objs['surf_loss'] = ch.concatenate([surf_loss[ids] *wt_surf*surf_wt_factor[k] for k, ids in can_meta['mrk_ids'].items()])
        # opt_objs['surf_loss'] = ch.concatenate([init_loss[ids] *wt_surf[k] for k, ids in can_meta['mrk_ids'].items()])

        if detailed_step:
            if len(pose_finger_ids):
                opt_objs['poseH'] = ch.concatenate([model.pose[pose_finger_ids] for model in opt_models]) * wt_poseH
            if len(pose_face_ids):
                opt_objs['poseF'] = ch.concatenate([model.pose[pose_face_ids] for model in opt_models]) * wt_poseF
                opt_objs['expr'] = ch.concatenate(v_face_exp) * wt_expr

            poses = pose_root_ids + pose_body_ids + pose_finger_ids + pose_face_ids
            if len(pose_body_ids) and not cfg.moshpp.optimize_toes:
                poses = list(set(poses).difference(set(pose_ids[30:36])))
            poses = [model.pose[poses] for model in opt_models]
            free_vars = poses + trans + v_face_exp + v_betas + [markers_latent]
        else:
            poses = pose_root_ids + pose_body_ids
            if len(pose_body_ids) and not cfg.moshpp.optimize_toes: poses = list(
                set(poses).difference(set(pose_ids[30:36])))
            poses = [model.pose[poses] for model in opt_models]
            free_vars = poses + trans + v_betas + [markers_latent]  # [tc.markers_latent[:len(markers_latent_B)]]

        logger.debug('Init. loss values: {}'.format(' | '.join(
            [f'{k} = {np.sum(opt_objs[k].r ** 2):2.2e}' for k in sorted(opt_objs)])))
        ch.minimize(fun=opt_objs, x0=free_vars,
                    callback=on_step,
                    method='dogleg',
                    options={'e_3': cfg.opt_settings.stagei_lr,
                             'delta_0': .5, 'disp': None,
                             'maxiter': cfg.opt_settings.maxiter})
        logger.debug("Final loss values: {}".format(' | '.join(
            [f'{k} = {np.sum(opt_objs[k].r ** 2):2.2e}' for k in sorted(opt_objs)])))

    # Saving all values from the optimization process, (sum of least squares)
    stagei_errs = {k: np.sum(obj_val.r ** 2) for k, obj_val in opt_objs.items()}

    sknbrs = NearestNeighbors(algorithm='kd_tree', n_neighbors=1).fit(can_model.r)
    _, closest = sknbrs.kneighbors(markers_latent.r)
    markers_latent_vids = {el[0]: el[1][0] for el in zip(latent_labels, closest.tolist())}

    all_mraker_locs = np.array([v.tolist() for v in stagei_frames[-1].values() if not np.any(np.isnan(v))])
    all_mraker_keys = np.array([k for k, v in stagei_frames[-1].items() if not np.any(np.isnan(v))])

    sknbrs = NearestNeighbors(algorithm='kd_tree', n_neighbors=1).fit(opt_models[-1].r)
    _, closest = sknbrs.kneighbors(all_mraker_locs)
    markers_latent_all_vids = {el[0]: el[1][0] for el in zip(all_mraker_keys, closest.tolist())}

    stagei_debug_details = {'opt_models_trans': [model.trans.r for model in opt_models],
                            'opt_models_pose': [model.pose.r for model in opt_models],
                            'stagei_errs': stagei_errs,
                            'markers_latent_all_vids': markers_latent_all_vids,
                            'stagei_markers_sim_all': [m.r for m in markers_sim_all],
                            'stagei_markers_sim': [m.r for m in markers_sim],
                            'stagei_markers_obs': [m for m in markers_obs],
                            'stagei_labels_obs': labels_obs,
                            }

    stagei_data = {'betas': can_model.betas.r if hasattr(can_model, 'betas') else None,
                   'markers_latent': markers_latent.r,
                   'latent_labels': latent_labels,
                   'marker_meta': marker_meta,
                   'markers_latent_vids': markers_latent_vids, }

    if v_template_fname is not None:
        stagei_data['v_template_fname'] = v_template_fname
        stagei_debug_details['v_template'] = can_model.v_template.r

    stagei_data['stagei_debug_details'] = stagei_debug_details

    return stagei_data


def mosh_stageii(mocap_fname: str, cfg: DictConfig, markers_latent: np.array,
                 latent_labels: list, betas: np.array, marker_meta: dict, v_template_fname=None) -> dict:
    num_train_markers = 46  # constant

    # 1. Load observed markers
    mocap = MocapSession(mocap_fname,
                         mocap_unit=cfg.mocap.unit,
                         mocap_rotate=cfg.mocap.rotate,
                         labels_map=general_labels_map,
                         # only_markers=latent_labels, # this is disable so that all point can appear in renders
                         )

    logger.debug('Loaded mocap markers for mosh stageii')

    can_model, opt_models = load_moshpp_models(surface_model_fname=cfg.surface_model.fname,
                                               surface_model_type=cfg.surface_model.type,
                                               num_beta_shared_models=1,  # one frame per opt step
                                               pose_hand_prior_fname=cfg.moshpp.pose_hand_prior_fname,
                                               pose_body_prior_fname=cfg.moshpp.pose_body_prior_fname,
                                               use_hands_mean=cfg.surface_model.use_hands_mean,
                                               dof_per_hand=cfg.surface_model.dof_per_hand,
                                               v_template_fname=v_template_fname)

    opt_model = opt_models[0]

    if hasattr(can_model, 'betas'):
        can_model.betas[:cfg.surface_model.num_betas] = betas[:cfg.surface_model.num_betas].copy()

    tc = TransformedCoeffs(can_body=can_model.r, markers_latent=markers_latent)
    markers_sim_all = TransformedLms(transformed_coeffs=tc, can_body=opt_model)

    logger.debug(f'#observed, #simulated markers: {len(mocap.labels)}, {len(markers_sim_all)}')

    if cfg.moshpp.optimize_dynamics:
        assert cfg.surface_model.type in ['smpl', 'smplh'], \
            NotImplementedError('DMPLs are currently only supported by smpl and smplh models')
        total_num_betas = cfg.surface_model.num_betas + cfg.surface_model.num_dmpls
        with open(cfg.surface_model.dmpl_fname) as f: dmpl_pcs = pickle.load(f)['eigvec']
        can_model.shapedirs[:, :, cfg.surface_model.num_betas:total_num_betas] = dmpl_pcs[:, :,
                                                                                 :cfg.surface_model.num_dmpls]
        opt_model.dmpl = opt_model.betas[cfg.surface_model.num_betas:total_num_betas]

    on_step = visualize_pose_estimate(opt_model, marker_meta=marker_meta) if cfg.moshpp.verbosity > 1 else None

    perframe_data = {
        'markers_sim': [],
        'markers_obs': [],
        'labels_obs': [],
        'fullpose': [],
        'trans': [],
        'stageii_errs': {},
    }
    if cfg.moshpp.optimize_dynamics: perframe_data['dmpls'] = []
    if cfg.moshpp.optimize_face: perframe_data['expression'] = []

    logger.debug(
        'mosh stageii weights are subject to change during the optimization, depending on how many markers are absent in each frame.')

    stageii_wts = cfg.opt_settings.weights
    logger.debug('MoSh stagei weights before annealing:\n{}'.format(
        '\n'.join(['{}: {}'.format(k, wt) for k, wt in stageii_wts.items() if k.startswith('stageii_wt')])))

    selected_frames = range(cfg.mocap.start_fidx, len(mocap) if cfg.mocap.end_fidx == -1 else cfg.mocap.end_fidx,
                            cfg.mocap.ds_rate)
    logger.debug(f'Starting mosh stageii for {len(selected_frames)} frames.')

    pose_prev = None
    dmpl_prev = None

    # Setup Variables
    v_face_exp = []

    pose_ids = list(range(opt_model.pose.size))
    pose_body_ids = []
    pose_face_ids = []
    pose_finger_ids = []
    pose_root_ids = pose_ids[:3]
    # v_pose_body = [model.pose[3:] for model in opt_models]
    if cfg.surface_model.type == 'smpl':
        pose_body_ids = pose_ids[3:]
    elif cfg.surface_model.type == 'smplh':
        pose_body_ids = pose_ids[3:66]
        if cfg.moshpp.optimize_fingers:  # dont chop chumpy variables two times
            pose_finger_ids = pose_ids[66:]
    elif cfg.surface_model.type == 'smplx':  # orient:3, body:63, jaw:3, eyel:3, eyer:3, handl, handr
        pose_body_ids = pose_ids[3:66]
        if cfg.moshpp.optimize_face:
            pose_face_ids = pose_ids[66:69]
            # pose_face_ids = pose_ids[66:75]
            v_face_exp = opt_model.betas[300:300 + cfg.surface_model.num_expressions]
        if cfg.moshpp.optimize_fingers:
            pose_finger_ids = pose_ids[75:]
    elif cfg.surface_model.type == 'mano':
        pose_finger_ids = pose_ids[3:]

    first_active_frame = True
    observed_markers_dict = mocap.markers_asdict()

    for fIdx, t in enumerate(selected_frames):

        if len(observed_markers_dict[t]) == 0:
            logger.error(f'no available observed markers for frame {t}. skipping the frame.')
            continue

        # Todo: should markers_obs be chumpy array?
        markers_obs = ch.vstack([observed_markers_dict[t][l] for l in latent_labels if l in observed_markers_dict[t]])
        markers_sim = ch.vstack(
            [markers_sim_all[lid] for lid, l in enumerate(latent_labels) if l in observed_markers_dict[t]])
        sim_labels = [l for l in latent_labels if l in observed_markers_dict[t]]

        num_missing_markers = float(len(markers_latent) - len(markers_sim))

        anneal_factor = 1.
        if num_missing_markers > 0:
            anneal_factor = anneal_factor + (
                    (num_missing_markers / len(markers_latent)) * stageii_wts.stageii_wt_annealing)

        wt_data = stageii_wts.stageii_wt_data * (num_train_markers / markers_obs.shape[0])
        wt_pose = stageii_wts.stageii_wt_poseB * anneal_factor
        wt_poseH = stageii_wts.stageii_wt_poseH * anneal_factor
        wt_poseF = stageii_wts.stageii_wt_poseF * anneal_factor
        wt_dmpl = stageii_wts.stageii_wt_dmpl
        wt_expr = stageii_wts.stageii_wt_expr
        wt_velo = stageii_wts.stageii_wt_velo

        # Setting up objective
        opt_objs = {'data': (markers_sim - markers_obs) * wt_data}
        if len(pose_body_ids):
            opt_objs['poseB'] = opt_model.priors['pose'](opt_model.pose[pose_body_ids]) * wt_pose

        # if len(pose_body_ids):# we dont have body for MANO
        #     opt_objs['poseB'] = ch.concatenate(opt_model.priors['pose'](opt_model.pose[pose_body_ids])) * wt_pose

        if pose_prev is not None:
            # extrapolating from prev 2 frames
            opt_objs['velo'] = (opt_model.pose - (opt_model.pose.r + (opt_model.pose.r - pose_prev))) * wt_velo

        # 1. Fit only the first frame
        if first_active_frame:  # np.median(np.abs(data_obj.r.ravel())) > .03:
            # Rigidly adjust poses/trans to fit bodies to landmarks
            logger.debug('Rigidly aligning the markers to the body...')
            # opt_model.pose[:] = opt_model.pose.r
            # opt_model.trans[:] = opt_model.trans.r
            perform_rigid_adjustment([opt_model.pose], [opt_model.trans], [opt_model], [markers_obs], [markers_sim])

            # for wt_pose_first in [5.]:
            for wt_pose_first in [10. * wt_pose, 5. * wt_pose, wt_pose]:
                if len(pose_body_ids):
                    opt_objs['poseB'] = opt_model.priors['pose'](opt_model.pose[pose_body_ids]) * wt_pose_first

                poses = pose_root_ids + pose_body_ids
                if len(pose_body_ids) and not cfg.moshpp.optimize_toes: poses = list(
                    set(poses).difference(set(pose_ids[30:36])))

                free_vars = [opt_model.trans, opt_model.pose[poses]]

                ch.minimize(fun=list(opt_objs.values()) if cfg.moshpp.verbosity == 0 else opt_objs, x0=free_vars,
                            method='dogleg',
                            options={'e_3': .001, 'delta_0': 5e-1, 'disp': None, 'maxiter': cfg.opt_settings.maxiter})

            first_active_frame = False
        else:
            pose_prev = opt_model.pose.r.copy()
            if cfg.moshpp.optimize_dynamics:
                dmpl_prev = opt_model.dmpl.r.copy()

        # 1. Warm start to correct pose
        logger.debug(
            f'{fIdx:04d}/{len(selected_frames):04d} -- Step 1. initial loss values: {" | ".join(["{} = {:2.2e}".format(k, np.sum(v.r ** 2)) for k, v in opt_objs.items()])}')

        poses = pose_root_ids + pose_body_ids
        if len(pose_body_ids) and not cfg.moshpp.optimize_toes: poses = list(
            set(poses).difference(set(pose_ids[30:36])))
        free_vars = [opt_model.trans, opt_model.pose[poses]]
        ch.minimize(fun=list(opt_objs.values()) if cfg.moshpp.verbosity == 0 else opt_objs, x0=free_vars,
                    method='dogleg',
                    options={'e_3': .01, 'delta_0': 5e-1, 'disp': None, 'maxiter': cfg.opt_settings.maxiter})
        logger.debug(
            f'{fIdx:04d}/{len(selected_frames):04d} -- Step 1. final loss values: {" | ".join(["{} = {:2.2e}".format(k, np.sum(v.r ** 2)) for k, v in opt_objs.items()])}')

        # 2. Fit for full pose
        free_vars = [opt_model.trans]
        current_v_pose_ids = pose_root_ids + pose_body_ids
        if len(pose_body_ids) and not cfg.moshpp.optimize_toes: current_v_pose_ids = list(
            set(current_v_pose_ids).difference(set(pose_ids[30:36])))

        if len(pose_finger_ids):
            opt_objs['poseH'] = opt_model.pose[pose_finger_ids] * wt_poseH
            current_v_pose_ids += pose_finger_ids

        if cfg.moshpp.optimize_face:
            opt_objs['poseF'] = opt_model.pose[pose_face_ids] * wt_poseF
            opt_objs['expr'] = v_face_exp * wt_expr
            free_vars += [v_face_exp]
            current_v_pose_ids += pose_face_ids
        free_vars += [opt_model.pose[current_v_pose_ids]]
        if cfg.moshpp.optimize_dynamics:
            if dmpl_prev is not None:
                opt_objs['extrap_dmpl'] = (opt_model.dmpl - (
                        opt_model.dmpl.r + (opt_model.dmpl.r - dmpl_prev))) * 6.0
            opt_objs['dmpl'] = opt_model.dmpl * wt_dmpl
            free_vars += [opt_model.dmpl]

        logger.debug(
            f'{fIdx:04d}/{len(selected_frames):04d} -- Step 2. initial loss values: {" | ".join(["{} = {:2.2e}".format(k, np.sum(v.r ** 2)) for k, v in opt_objs.items()])}')
        ch.minimize(fun=list(opt_objs.values()) if cfg.moshpp.verbosity == 0 else opt_objs, x0=free_vars,
                    method='dogleg',
                    options={'e_3': .01, 'delta_0': 5e-1, 'disp': None, 'maxiter': cfg.opt_settings.maxiter})
        logger.debug(
            f'{fIdx:04d}/{len(selected_frames):04d} -- Step 2. final loss values: {" | ".join(["{} = {:2.2e}".format(k, np.sum(v.r ** 2)) for k, v in opt_objs.items()])}')

        if on_step: on_step(markers_obs=markers_obs.r, markers_sim=markers_sim.r, sim_labels=sim_labels,
                            fIdx=t)  # show shape after pre-setup pose estimation step

        for k, v in opt_objs.items():
            if k not in perframe_data['stageii_errs']: perframe_data['stageii_errs'][k] = []
            perframe_data['stageii_errs'][k].append(np.sum(v.r ** 2))

        perframe_data['markers_sim'].append(markers_sim.r.copy())
        perframe_data['markers_obs'].append(markers_obs.r.copy())
        perframe_data['labels_obs'].append([l for l in latent_labels if l in observed_markers_dict[t]])
        perframe_data['fullpose'].append(opt_model.fullpose.r.copy())
        perframe_data['trans'].append(opt_model.trans.r.copy())
        if cfg.moshpp.optimize_dynamics:
            perframe_data['dmpls'].append(opt_model.betas.r[cfg.surface_model.num_betas:total_num_betas].copy())
        if cfg.moshpp.optimize_face:
            perframe_data['expression'].append(opt_model.betas[300:].r.copy())

    stageii_debug_details = {
        'stageii_errs': {k: np.array(v) for k, v in perframe_data.pop('stageii_errs').items()},
        'markers_sim': perframe_data.pop('markers_sim'),
        'markers_obs': perframe_data.pop('markers_obs'),
        'labels_obs': perframe_data.pop('labels_obs'),
        'markers_orig': mocap.markers[selected_frames],
        'labels_orig': mocap.labels,
        'mocap_fname': mocap_fname,
        'mocap_frame_rate': mocap.frame_rate,
        'mocap_time_length': mocap.time_length(),

    }
    stageii_data = {k: np.array(v) for k, v in perframe_data.items()}
    stageii_data['stageii_debug_details'] = stageii_debug_details

    return stageii_data
