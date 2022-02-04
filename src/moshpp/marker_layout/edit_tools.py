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
import json
import os
import sys
from collections import OrderedDict
from os import path as osp
from pathlib import Path
from typing import OrderedDict as OrderedDictType
from typing import Union, List

import numpy as np
import scipy.sparse as sp
import torch
from body_visualizer.tools.vis_tools import colors
from colour import Color
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import makepath
from human_body_prior.tools.rotation_tools import rotate_points_xyz
from loguru import logger
from psbody.mesh import Mesh

from moshpp.marker_layout.labels_map import general_labels_map
from moshpp.tools.mocap_interface import write_mocap_c3d

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

from typing import Dict
from pytorch3d.structures import Meshes
from human_body_prior.tools.omni_tools import get_support_data_dir


class Markerlayout(TypedDict):
    marker_vids: OrderedDictType[str, int]
    marker_type: OrderedDictType[str, str]
    marker_type_mask: OrderedDictType[str, bool]
    m2b_distance: OrderedDictType[str, float]
    surface_model_type: str
    marker_colors: OrderedDictType[str, float]
    marker_layout_fname: str


class SuperSet(Markerlayout):
    marker_vids: OrderedDictType[str, List[int]]
    marker_layout_fname: str = None


def marker_layout_load(marker_layout_fname: Union[str, Path],
                       labels_map: Dict[str, str] = general_labels_map,
                       include_nan: bool = True,
                       exclude_marker_types: List[str] = None,
                       exclude_markers: List[str] = None,
                       only_markers: List[str] = None,
                       verbosity: int = 1,
                       ) -> Markerlayout:
    """
    :param marker_layout_fname: a json file for the marker layout data
    :param labels_map:
    :param include_nan:
    :param exclude_markers:
    :param exclude_marker_types:
    :param only_markers:
    :return:
        marker_vids: an ordered dictionary holding label: vid (integer).
                    the order is first based on marker type then based on label.
                    if a label mapping is presented the ordering is done after applying the mapping
        labels_color: an ordered dictionary holding label: color (rgb values as floats)
        marker_type_mask: a dictionary holding the indices of different marker types in marker_vids.keys(): marker_type: list (booleans)
        m2b_distance: a dictionary holding the distance_from_skin value of each marker type in marker types: marker_type: distance (float)
        surface_model_type: the type of the surface model that this marker layout is suitable for
    """

    assert marker_layout_fname.endswith('.json')
    assert osp.exists(marker_layout_fname), FileNotFoundError(marker_layout_fname)

    if only_markers is None: only_markers = []
    if exclude_markers is None: exclude_markers = []
    if exclude_marker_types is None: exclude_marker_types = []

    with open(marker_layout_fname) as f:
        d = json.load(f)

    marker_vids = OrderedDict()  # an ordered dictionary of label:vid
    marker_types = OrderedDict()  # a dictionary with marker_type:[marker labels]
    m2b_distance = OrderedDict()  #

    if 'surface_model_type' not in d:
        logger.debug(
            f'Assuming SMPLx for marker layout since surface_model_type field was not available in: {marker_layout_fname}')
        surface_model_type = 'smplx'
    else:
        surface_model_type = d['surface_model_type']

    if verbosity > 0:
        logger.info(f'Loading marker layout: {marker_layout_fname}')
        logger.info('Available marker types: {}. Total: {} markers.'.format(
            {markerset['type']: len(markerset['indices']) for markerset in
             sorted(d['markersets'], key=lambda a: a['type'])},
            sum([len(markerset['indices']) for markerset in d['markersets']])))

    for markerset in sorted(d['markersets'], key=lambda a: a['type']):
        marker_type = markerset['type']
        if marker_type in exclude_marker_types:
            logger.debug(f'excluding marker_type {marker_type}')
            continue
        if marker_type in m2b_distance: raise ValueError(
            f'Marker type appears in multiple occasions: {markerset["type"]}!')
        m2b_distance[marker_type] = markerset.get('distance_from_skin', 0.0095)
        cur_marker_vids = markerset['indices']
        if labels_map:
            cur_marker_vids = {labels_map.get(k, k): cur_marker_vids[k] for k in cur_marker_vids}

        for label in sorted(cur_marker_vids):
            if only_markers and label not in only_markers: continue
            if label in exclude_markers:
                logger.debug(f'excluding label {label}')
            vid = cur_marker_vids[label]
            if label in marker_vids: raise ValueError(f'Label ({label}) is present in multiple occasions.')
            marker_vids[label] = vid
            if markerset['type'] not in marker_types: marker_types[markerset['type']] = []
            if labels_map:
                marker_types[markerset['type']].append(labels_map.get(label, label))
            else:
                marker_types[markerset['type']].append(label)

    marker_type_mask = OrderedDict(
        {k: np.array([True if l in marker_types[k] else False for l in marker_vids.keys()]) for k in marker_types})
    marker_colors = OrderedDict(
        {k: v.get_rgb() for k, v in zip(marker_vids, list(Color('red').range_to(Color('blue'), len(marker_vids))))})

    if include_nan: marker_colors['nan'] = [0.83, 1, 0]  # yellow green

    marker_type = OrderedDict()
    for lid, l in enumerate(marker_vids):
        for cur_marker_type, type_mask in marker_type_mask.items():
            if type_mask[lid]:
                marker_type[l] = cur_marker_type
                continue

    out_marker_layout: Markerlayout = {'marker_vids': marker_vids,
                                       'marker_colors': marker_colors,
                                       'marker_type': marker_type,
                                       'marker_type_mask': marker_type_mask,
                                       'm2b_distance': m2b_distance,
                                       'surface_model_type': surface_model_type,
                                       'marker_layout_fname': marker_layout_fname}

    return out_marker_layout


def marker_meta_filter(marker_meta: Markerlayout, interested_labels: List[str]):
    import copy

    new_meta = copy.deepcopy(marker_meta)

    available_mask = [l in interested_labels for l in marker_meta['marker_vids'].keys()]

    for marker_type, marker_mask in new_meta['marker_type_mask'].items():
        new_meta['marker_type_mask'][marker_type] = (np.array(marker_mask)[available_mask]).tolist()

    new_meta['marker_vids'] = OrderedDict(
        {k: v for k, v in marker_meta['marker_vids'].items() if k in interested_labels})
    new_meta['marker_colors'] = OrderedDict(
        {k: v for k, v in marker_meta['marker_colors'].items() if k in interested_labels + ['nan']})
    return new_meta


def marker_layout_write(marker_meta: Markerlayout, marker_layout_fname: Union[str, Path]) -> None:
    assert marker_layout_fname.endswith('.json')
    makepath(marker_layout_fname, isfile=True)

    marker_layout = {'surface_model_type': marker_meta['surface_model_type'], 'markersets': []}
    for marker_type, marker_mask in marker_meta['marker_type_mask'].items():
        marker_layout['markersets'].append({
            'indices': {
                l: [int(vid) for vid in marker_meta['marker_vids'][l]] if isinstance(marker_meta['marker_vids'][l],
                                                                                     list) else int(
                    marker_meta['marker_vids'][l]) for l in
                np.array(list(marker_meta['marker_vids'].keys()))[marker_mask]},
            'distance_from_skin': marker_meta['m2b_distance'][marker_type],
            'type': marker_type
        })

    with open(marker_layout_fname, 'w') as f:
        json.dump(marker_layout, f, sort_keys=True, indent=2, separators=(',', ': '))


def merge_marker_layouts(marker_layout_fnames: List[Union[str, Path]],
                         out_fname: Union[str, Path] = None,
                         labels_map: Dict[str, str] = general_labels_map) -> SuperSet:
    def flatten_list(list_d):
        out_list = []
        for a in list_d:
            if isinstance(a, list):
                out_list.extend(a)
            else:
                out_list.append(a)
        return out_list

    assert len(marker_layout_fnames) != 0

    if out_fname is None or not os.path.exists(out_fname):
        logger.debug(f'Merging #{len(marker_layout_fnames)} marker layouts: {marker_layout_fnames}.')
    else:
        logger.debug(f'Superset file already exists at {out_fname}')
        return marker_layout_load(out_fname, labels_map=general_labels_map)

    marker_vids = {}
    m2b_distance = {}
    surface_model_types = {}
    for marker_layout_fname in marker_layout_fnames:
        marker_meta = marker_layout_load(marker_layout_fname, labels_map=labels_map)
        surface_model_types[marker_layout_fname] = marker_meta['surface_model_type']
        for marker_type, marker_mask in marker_meta['marker_type_mask'].items():
            # if marker_type != 'body': raise ValueError(marker_layout_fname)
            if marker_type not in marker_vids: marker_vids[marker_type] = {}
            for k, v, is_in_marker_type in zip(marker_meta['marker_vids'].keys(),
                                               marker_meta['marker_vids'].values(),
                                               marker_mask):
                if not is_in_marker_type: continue
                if k not in marker_vids[marker_type]: marker_vids[marker_type][k] = []
                marker_vids[marker_type][k].append(v)
            if marker_type in m2b_distance: assert m2b_distance[marker_type] == marker_meta['m2b_distance'][marker_type]
            m2b_distance[marker_type] = marker_meta['m2b_distance'][marker_type]

    assert len(set(surface_model_types.values())) == 1, ValueError(
        f'Marker layout of multiple surface types cannot be merged: {surface_model_types}')

    flattened_marker_vids = {k: list(set(flatten_list(v))) for mrk_type in marker_vids.keys() for k, v in
                             marker_vids[mrk_type].items()}
    marker_type_mask = {marker_type: [True if l in marker_type_labels else False for l in flattened_marker_vids] for
                        marker_type, marker_type_labels in marker_vids.items()}

    marker_type = OrderedDict()
    for lid, l in enumerate(marker_vids):
        for cur_marker_type, type_mask in marker_type_mask.items():
            if type_mask[lid]:
                marker_type[l] = cur_marker_type
                continue

    new_marker_meta: SuperSet = {'marker_vids': flattened_marker_vids,
                                 'marker_type': marker_type,
                                 'marker_type_mask': marker_type_mask,
                                 'm2b_distance': m2b_distance,
                                 'surface_model_type': list(set(surface_model_types.values()))[0]
                                 }
    if out_fname is not None:
        marker_layout_write(new_marker_meta, out_fname)
        logger.debug(f'Merged marker layout file created at {out_fname}.')
    return new_marker_meta


def marker_layout_as_mesh(surface_model_fname: Union[str, Path],
                          body_parms: dict = {},
                          ceasar_pose: bool = False,
                          preserve_vertex_order: bool = False,
                          surface_model_type=None):
    marker_radius = {'body': 0.009, 'face': 0.004, 'finger': 0.005}

    if ceasar_pose:
        support_base_dir = get_support_data_dir(__file__)
        pose_body = np.load(osp.join(support_base_dir, 'smplx_APose.npz'))['pose_body']
        pose_body_pt = torch.from_numpy(pose_body.reshape(1, -1)).type(torch.float)
        body_parms = {'pose_body': pose_body_pt}

    bm = BodyModel(surface_model_fname, num_betas=body_parms.get('num_betas', 10), model_type=surface_model_type)
    body = bm(**body_parms)

    dtvn = c2c(Meshes(verts=body.v, faces=body.f.expand(len(body.v), -1, -1)).verts_normals_packed())

    verts = c2c(body.v[0])
    faces = c2c(body.f)

    if preserve_vertex_order:
        from body_visualizer.mesh.psbody_mesh_sphere import points_to_spheres
        body_mesh = Mesh(verts, faces, vc=colors['grey'])
        # dtvn = body_mesh.estimate_vertex_normals()

    else:
        from body_visualizer.mesh.sphere import points_to_spheres
        import trimesh
        body_mesh = trimesh.Trimesh(vertices=verts, faces=faces,
                                    vertex_colors=np.tile(colors['grey'], (verts.shape[0], 1)))
        # dtvn = Mesh(verts, faces, vc=colors['grey']).estimate_vertex_normals()

    def as_mesh(marker_layout_fname: Union[str, Path],
                out_fname: Union[str, Path] = None,
                marker_colors: OrderedDictType[str, float] = None):
        output = {}
        if isinstance(marker_layout_fname, dict):
            marker_meta = marker_layout_fname
        else:
            marker_meta = marker_layout_load(marker_layout_fname)

        assert bm.model_type == marker_meta['surface_model_type'], \
            ValueError(f'model_type of body model and marker layout are not equal: '
                       f'{bm.model_type} != {marker_meta["surface_model_type"]}')

        if marker_colors is not None:
            for l in marker_meta['marker_colors']:
                if l not in marker_colors: raise ValueError(f'No color available in marker_colors for label: {l}')

        marker_m2b = np.ones(len(marker_meta['marker_vids'])) * 0.0095
        marker_radi = np.ones(len(marker_meta['marker_vids'])) * marker_radius['body']
        for marker_type, mask in marker_meta['marker_type_mask'].items():
            marker_m2b[mask] = marker_meta['m2b_distance'][marker_type]
            k = [k for k in marker_radius if k in marker_type]
            if len(k) == 1:
                marker_radi[mask] = marker_radius[k[0]]
            else:
                print('No radius found for {}'.format(marker_type))
                continue

        marker_colors = list(marker_meta['marker_colors'].values())[:-1] if marker_colors is None else \
            [marker_colors[l] for l in marker_meta['marker_colors']]

        sample_value = list(marker_meta['marker_vids'].values())[0]
        is_superset = isinstance(sample_value, list)
        body_vids = [np.random.choice(vids).tolist() for vids in marker_meta['marker_vids'].values()] if is_superset \
            else list(marker_meta['marker_vids'].values())

        markers = verts[body_vids] + dtvn[body_vids] * np.array(marker_m2b)[:, None]

        if preserve_vertex_order:
            markers_mesh = points_to_spheres(markers, point_color=np.array(marker_colors), radius=marker_radi)
            body_mrk_mesh = body_mesh.concatenate_mesh(markers_mesh)
        else:
            markers_mesh = points_to_spheres(markers, point_color=marker_colors, radius=marker_radi)
            body_mrk_mesh = trimesh.util.concatenate([body_mesh, markers_mesh])

            output.update({
                'body_mesh_unrotated': body_mesh,
                'markers_mesh_unrotated': markers_mesh,
            })
            body_mrk_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90), (1, 0, 0)))
            markers_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90), (1, 0, 0)))
            body_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90), (1, 0, 0)))

        if out_fname is not None:
            assert out_fname.endswith('.ply'), ValueError(f'out_fname should be a valid ply file: {out_fname}')
            if preserve_vertex_order:
                body_mrk_mesh.write_ply(out_fname)
            else:
                body_mrk_mesh.export(out_fname)
                import sys
                sys.stderr.write('Mesh exported via Trimesh. Export does not preserve vertex orders!!!!!\n')

        output.update({
            'body_marker_mesh': body_mrk_mesh,
            'body_mesh': body_mesh,
            'markers_mesh': markers_mesh,
            'markers': markers, 'labels': list(marker_meta['marker_vids'].keys())})
        return output

    return as_mesh


def marker_layout_to_c3d(marker_layout_fname, surface_model_fname, out_c3d_fname=None):
    """
    This function enables investigating a marker layout in c3d software such as Mokka
    Args:
        marker_layout_fname:
        surface_model_fname:
        out_c3d_fname:

    Returns:

    """
    if out_c3d_fname is None:
        out_c3d_fname = marker_layout_fname.replace('.json', '.c3d')
    assert out_c3d_fname.endswith('.c3d'), ValueError(f'out_c3d_fname should be a valid c3d file {out_c3d_fname}')

    mocap = marker_layout_as_mesh(surface_model_fname)(marker_layout_fname)
    markers = mocap['markers'] + [0, 1.3, 0]
    markers = rotate_points_xyz(markers[None], np.array([90, 0, 0])[None])
    markers = np.repeat(markers, repeats=100, axis=0)

    write_mocap_c3d(markers=markers,
                    labels=mocap['labels'],
                    out_mocap_fname=out_c3d_fname,
                    frame_rate=60)


def find_vertex_neighbours(surface_model_fname):
    from human_body_prior.body_model.body_model import BodyModel
    from human_body_prior.tools.omni_tools import copy2cpu as c2c

    sm = BodyModel(surface_model_fname)
    A = get_vert_connectivity(c2c(sm().v[0]), c2c(sm.f))

    A = np.asarray(A.todense())

    def get_neighbour(vid, n_ring=1):
        """

        Args:
            vid: int
            n_ring:

        Returns:
            list of integers

        """
        if n_ring == 0: return [vid]  # randomiztion is disabled
        neighbors = [(np.arange(A.shape[0])[A[vid] > 0]).tolist()]
        for _ in range(n_ring - 1):
            new_neighbors = []
            for vid in neighbors[-1]:
                # if vid in disabled_vids: continue
                new_neighbors.extend((np.arange(A.shape[0])[A[vid] > 0]).tolist())
            neighbors.append(new_neighbors)
        flat_list = [item for sublist in neighbors for item in sublist]
        flat_list = list(set(flat_list))
        return flat_list

    return get_neighbour


def row(A):
    return A.reshape((1, -1))


def col(A):
    return A.reshape((-1, 1))


def get_vert_connectivity(mesh_v, mesh_f):
    """Returns a sparse matrix (of size #verts x #verts) where each nonzero
    element indicates a neighborhood relation. For example, if there is a
    nonzero element in position (15,12), that means vertex 15 is connected
    by an edge to vertex 12."""

    vpv = sp.csc_matrix((len(mesh_v), len(mesh_v)))

    # for each column in the faces...
    for i in range(3):
        IS = mesh_f[:, i]
        JS = mesh_f[:, (i + 1) % 3]
        data = np.ones(len(IS))
        ij = np.vstack((row(IS.ravel()), row(JS.ravel())))
        mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
        vpv = vpv + mtx + mtx.T

    return vpv


def randomize_marker_layout_vids(marker_vids,
                                 marker_type_mask,
                                 surface_model_fname,
                                 n_ring=1,
                                 enable_rnd_vid_on_face_hands=True):
    """

    Args:
        marker_vids: <key: str, value: str/list>
        n_ring: integer indicating the neighbourhood ring

    Returns:

    """
    from human_body_prior.tools.omni_tools import flatten_list

    assert osp.exists(surface_model_fname)
    v_neighbors = find_vertex_neighbours(surface_model_fname)

    sample_value = list(marker_vids.values())[0]
    is_superset = isinstance(sample_value, list)

    if is_superset:
        if enable_rnd_vid_on_face_hands:
            vid_neighbours = {k: flatten_list([v_neighbors(vid, n_ring=n_ring) for vid in vids]) + vids for k, vids in
                              marker_vids.items()}
        else:
            assert 'body' in marker_type_mask, ValueError(f'body not available marker_types: {marker_type_mask.keys()}')
            vid_neighbours = {
                k: flatten_list([v_neighbors(vid, n_ring=n_ring) for vid in vids]) + vids if isbody else vids for
                (k, vids), isbody in zip(marker_vids.items(), marker_type_mask['body'])}
    else:
        if enable_rnd_vid_on_face_hands:
            vid_neighbours = {k: v_neighbors(vid, n_ring=n_ring) + [vid] for k, vid in marker_vids.items()}
        else:
            assert 'body' in marker_type_mask, ValueError(f'body not available marker_types: {marker_type_mask.keys()}')
            vid_neighbours = {k: v_neighbors(vid, n_ring=n_ring) + [vid] if isbody else [vid] for (k, vid), isbody in
                              zip(marker_vids.items(), marker_type_mask['body'])}

    def get_next():
        new_marker_vids = OrderedDict()
        for k, vid in marker_vids.items():
            new_marker_vids[k] = np.random.choice(vid_neighbours[k]).tolist()
        return new_marker_vids

    return get_next

