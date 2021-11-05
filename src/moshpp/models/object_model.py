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
import chumpy as ch
import numpy as np
from psbody.smpl.rodrigues import Rodrigues


class RigidObjectModel(ch.Ch):
    dterms = ['trans', 'pose']

    def __init__(self, ply_fname):
        self.trans = ch.asarray(np.zeros((3), dtype=np.float32))
        self.pose = ch.asarray(np.zeros((3), dtype=np.float32))

        rigid_mesh = Mesh(filename=ply_fname)
        rigid_v = rigid_mesh.v
        self.f = rigid_mesh.f

        self.v = ch.dot(rigid_v, Rodrigues(self.pose)) + self.trans

    def compute_r(self):
        return self.v.r

    def compute_dr_wrt(self, wrt):
        return self.v.dr_wrt(wrt)


if __name__ == '__main__':
    obj_model = RigidObjectModel(
        ply_fname='/ps/scratch/body_hand_object_contact/data/object_settings/flute_smaller.ply')

    print(obj_model.dr_wrt(obj_model.trans).shape)
    obj_model.trans[:] = np.ones_like(obj_model.trans)
    print(obj_model.dr_wrt(obj_model.trans).shape)

    from psbody.mesh import Mesh

    Mesh(v=obj_model.v, f=obj_model.f).show()
