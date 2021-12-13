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
from functools import reduce

import numpy as np
from body_visualizer.mesh.psbody_mesh_sphere import points_to_spheres
from psbody.mesh.lines import Lines
from psbody.mesh.mesh import Mesh
from psbody.mesh.meshviewer import MeshViewer, MeshViewers

frame_num = 0
mvs = None
mv_canonical = None
mv2 = None

marker_radius = {'body': 0.009, 'face': 0.004, 'finger': 0.005, 'finger_left': 0.005, 'finger_right': 0.005}


def visualize_shape_estimate(opt_models, can_model, markers_sim, markers_obs, markers_latent, init_markers_latent,
                             marker_meta):
    from psbody.mesh.meshviewer import test_for_opengl
    if not test_for_opengl(): return

    marker_radi = np.ones(len(marker_meta['marker_vids'])) * marker_radius['body']
    for marker_type, mask in marker_meta['marker_type_mask'].items():
        marker_radi[mask] = marker_radius.get(marker_type, marker_radius['body'])

    sz = int(np.ceil(np.sqrt(len(opt_models))))

    mvs_raw = MeshViewers(window_width=640, window_height=480, shape=(sz, sz))
    mvs = reduce(lambda x, y: x + y, mvs_raw)
    mv_canonical = MeshViewer(window_width=640, window_height=480, keepalive=False)
    mv_canonical.set_background_color(np.array([1., 1., 1.]))
    mvs[0].set_background_color(np.array([1., 1., 1.]))

    # Callback during optimization, to show what's happening
    def on_step(_):

        for model_idx, model in enumerate(opt_models):
            linev = np.hstack((markers_obs[model_idx], markers_sim[model_idx])).reshape((-1, 3))
            linee = np.arange(len(linev)).reshape((-1, 2))
            ll = Lines(v=linev, e=linee)
            ll.vc = (ll.v * 0. + 1) * np.array([0.00, 0.00, 1.00])

            mvs[model_idx].static_meshes = [
                Mesh(v=opt_models[model_idx].r, f=opt_models[model_idx].f, vc=opt_models[model_idx].r * 0 + .7)]
            mvs[model_idx].static_lines = [ll]

        mv_canonical.set_static_meshes([Mesh(v=can_model.r, f=can_model.f, vc=can_model.r * 0 + .7)])
        # print(init_markers_latent, np.sum(np.square(init_markers_latent- markers_latent)))
        # print(init_markers_latent.r[0], init_markers_latent.r[-1], tc.r[0], tc.r[-1])
        init_markers_latent_mesh = points_to_spheres(np.asarray(init_markers_latent),
                                                     point_color=np.array((0., 0., 1.)), radius=marker_radi)
        markers_latent_mesh = points_to_spheres(markers_latent.r, point_color=np.array((1., 0., 0.)),
                                                radius=marker_radi)

        mv_canonical.set_dynamic_meshes([init_markers_latent_mesh, markers_latent_mesh], blocking=True)  # red + blue

    return lambda x: on_step(x)


def visualize_pose_estimate(sv, marker_meta):
    from psbody.mesh.meshviewer import test_for_opengl
    if not test_for_opengl(): return

    marker_radi = {}
    for marker_type, mask in marker_meta['marker_type_mask'].items():
        for valid, l in zip(mask, list(marker_meta['marker_vids'].keys())):
            if valid: marker_radi[l] = marker_radius.get(marker_type, marker_radius['body'])

    mv2 = MeshViewer(window_width=640, window_height=480, keepalive=False)

    # Callback during optimization, to show what's happening
    def on_step(markers_obs, markers_sim, sim_labels, fIdx):

        linev = np.hstack((markers_obs, markers_sim)).reshape((-1, 3))
        linee = np.arange(len(linev)).reshape((-1, 2))
        ll = Lines(v=linev, e=linee)
        ll.vc = (ll.v * 0. + 1) * np.array([0., .5, .6])

        mv2.static_meshes = [Mesh(v=sv.r, f=sv.f)]
        mv2.static_lines = [ll]

        radius = [marker_radi[l] for l in sim_labels]

        markers_obs_spheres = points_to_spheres(markers_obs, point_color=np.array((0., 0., 1.)), radius=radius)  # blue
        markers_sim_spheres = points_to_spheres(markers_sim, point_color=np.array((1., 0., 0.)), radius=radius)  # blue

        mv2.set_dynamic_meshes([markers_sim_spheres, markers_obs_spheres], blocking=True)  # red + blue

    return on_step
