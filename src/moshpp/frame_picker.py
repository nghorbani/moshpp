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
from typing import List

import numpy as np

from moshpp.tools.mocap_interface import MocapSession


def load_marker_sessions_manual(mocap_fnames: List[str], mocap_unit: str, mocap_rotate: list = None,
                                only_subjects: List[str] = None,
                                only_markers=None, exclude_markers=None, labels_map={}):
    """
    in manual mode the stagei_mocap_fnames should be /path/to/mocap_framenumer.extention
    """
    all_frames = []
    all_fnames = []
    for frame in mocap_fnames:
        frame_splits = frame.split('_')
        frame_fname, frame_id = '_'.join(frame_splits[:-1]), int(frame_splits[-1])
        assert osp.exists(frame_fname), FileNotFoundError(frame_fname)
        key = f'{frame_fname}_{frame_id:06d}'
        all_fnames.append(key)
        all_frames.append(MocapSession(mocap_fname=frame_fname,
                                       mocap_unit=mocap_unit,
                                       only_subjects=only_subjects,
                                       mocap_rotate=mocap_rotate,
                                       only_markers=only_markers,
                                       exclude_markers=exclude_markers,
                                       labels_map=labels_map).markers_asdict()[frame_id])

    all_frames = np.array(all_frames)
    all_fnames = np.array(all_fnames)

    return all_frames, all_fnames


def load_marker_sessions_random(mocap_fnames: List[str], mocap_unit: str, mocap_rotate: list = None,
                                num_frames: int = 12,
                                only_subjects: List[str] = None,
                                seed: int = None, least_avail_markers: float = .1,
                                only_markers=None, exclude_markers=None, labels_map={}):
    """
    Randomly select num_frames number of frames from the set of mocap files given.
    the chosen frames should include more than least_avail_markers of their respective markers
    if not enough frames are found least_avail_markers is lowered

    :param mocap_fnames:
    :param mocap_unit:
    :param mocap_rotate: 
    :param only_markers:
    :param num_frames:
    :param seed:
    :param least_avail_markers: at least this percentage of mocap markers should be present at selected frames
    :param exclude_markers:
    :param labels_map:
    :return:
    """

    fname_frame_to_markers = {}
    for fname in mocap_fnames:
        mocap = MocapSession(mocap_fname=fname, mocap_unit=mocap_unit, mocap_rotate=mocap_rotate,
                             only_subject=only_subjects,
                             only_markers=only_markers, exclude_markers=exclude_markers, labels_map=labels_map)
        # Creating a dict of filename_frame# to observed markers so that we can refer back to this dict
        # if we need to find which file & frame number each observed marker-frame came from
        frame_marker_dict = mocap.markers_asdict()
        frame_marker_dict = [frame_marker_dict[i] for i in np.random.choice(len(mocap), num_frames)]
        for fIdx in range(len(frame_marker_dict)):
            key = f'{fname}_{fIdx:06d}'
            fname_frame_to_markers[key] = frame_marker_dict[fIdx]
        # adding a break in the loop when reading all session files when the process has read more than xxx frames
        if len(fname_frame_to_markers.keys()) > 100:
            break

    idxs_to_shuffle = list(range(len(fname_frame_to_markers.keys())))
    # Shuffle frame indices
    if seed is None:
        # undeterministic
        np.random.shuffle(idxs_to_shuffle)
    else:
        np.random.seed(seed=seed)
        np.random.shuffle(idxs_to_shuffle)

    all_frames = list(fname_frame_to_markers.values())
    all_fnames = list(fname_frame_to_markers.keys())

    reordered_frames = []
    reordered_fnames = []
    for idx in idxs_to_shuffle:
        frame = all_frames[idx]
        fname = all_fnames[idx]
        # removing frames with no more than ...% markers available
        nonans = [k for k in frame.keys() if ~np.any(np.isnan(frame[k])) and ('*' not in k)]
        if len(nonans) >= (least_avail_markers * len(frame)):
            reordered_fnames.append(fname)
            reordered_frames.append(frame)
        if len(reordered_frames) >= num_frames: break

    reordered_frames = np.array(reordered_frames)
    reordered_fnames = np.array(reordered_fnames)
    # assert(len(reordered_frames)>=num_frames)
    if len(reordered_frames) < num_frames:
        least_avail_markers = least_avail_markers - 0.01
        if least_avail_markers < 0.01:
            raise ValueError(
                f'Not enough frames were found that have at least %{least_avail_markers * 100.:.1f} of the markers.\n')

        return load_marker_sessions_random(mocap_fnames, mocap_unit=mocap_unit, mocap_rotate=mocap_rotate, seed=seed,
                                           num_frames=num_frames, only_subject=only_subjects,
                                           least_avail_markers=least_avail_markers, only_markers=only_markers,
                                           labels_map=labels_map)
    return reordered_frames, reordered_fnames


def load_marker_sessions_random_strict(mocap_fnames: List[str], mocap_unit: str, mocap_rotate: list = None,
                                       num_frames: int = 12,
                                       only_subjects: List[str] = None,
                                       seed: int = None, least_avail_markers: float = .1,
                                       only_markers=None, exclude_markers=None, labels_map={}):
    """
    will randomly select num_frames number of frames from the set of mocap files given.
    the chosen frames should include more than xx of their respective markers or given labels
    this function will not lower the marker availability threshold and will raise an error if not enough frames available

    :param mocap_fnames:
    :param mocap_unit:
    :param mocap_rotate:
    :param only_markers:
    :param num_frames:
    :param seed:
    :param least_avail_markers: at least this percentage of mocap labels should be present at selected frames
    :param exclude_markers:
    :param labels_map:
    :return:
    """
    np.random.seed(seed=seed)
    assert least_avail_markers >= 0.1 and least_avail_markers <= 1.0

    fname_frame_to_markers = {}
    for fname in mocap_fnames:
        mocap = MocapSession(mocap_fname=fname,
                             mocap_unit=mocap_unit,
                             mocap_rotate=mocap_rotate,
                             only_markers=only_markers,
                             only_subjects=only_subjects,
                             exclude_markers=exclude_markers,
                             labels_map=labels_map)
        marker_availability_norm = MocapSession.marker_availability_mask(mocap.markers)
        marker_availability_norm = marker_availability_norm.sum(-1) / marker_availability_norm.shape[1]

        # Creating a dict of filename_frame# to observed markers so that we can refer back to this dict
        # if we need to find which file & frame number each observed marker-frame came from
        frames = mocap.markers_asdict()
        # fids = range(len(mocap))
        # np.random.shuffle(fids)
        cur_mocap_n_picks = 0
        for fIdx in np.random.choice(len(frames), len(frames), replace=False):
            key = f'{fname}_{fIdx:06d}'
            if marker_availability_norm[fIdx] >= least_avail_markers:
                fname_frame_to_markers[key] = frames[fIdx]
                cur_mocap_n_picks += 1
            if cur_mocap_n_picks >= num_frames: break

        # adding a break in the loop when reading all session files when the process has read more than xxx frames
        if len(fname_frame_to_markers.keys()) > 100:
            break

    if len(fname_frame_to_markers) < num_frames:
        raise ValueError(
            f'Not enough frames were found that have at least {least_avail_markers * 100.:.1f}% of the markers.\n')

    ids = np.random.choice(len(fname_frame_to_markers), num_frames, replace=False)

    all_frames = np.array(list(fname_frame_to_markers.values()))[ids]
    all_fnames = np.array(list(fname_frame_to_markers.keys()))[ids]

    return all_frames, all_fnames
