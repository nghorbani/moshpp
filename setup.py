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

from pathlib import Path
import platform
from setuptools import Extension, find_packages, setup
import sysconfig

import numpy

PACKAGE = 'moshpp'

# mesh distance python module
cython_file = Path(__file__).parent / 'moshpp' / 'scan2mesh' / \
    'mesh_distance' / 'sample2meshdist.pyx'
sourcefiles = [str(cython_file.resolve())]
additional_options = {'include_dirs': [numpy.get_include(), '/usr/local/include']}

if platform.system().lower() in ['darwin', 'linux']:

    extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
    extra_compile_args += ["-std=c++11"]
    additional_options['extra_compile_args'] = extra_compile_args


def _get_version():
    """"Helper to get the package version."""

    version_path = Path() / PACKAGE / 'version.py'
    if not version_path.exists:
        return None
    with open(version_path, 'r') as version_file:
        ns = {}
        exec(version_file.read(), ns)
    return ns['__version__']


dependencies = [
    'chumpy',
    'opencv-python',
    'matplotlib',
    'sklearn',
    'numpy',
    'loguru',
    'cython'
]

setup(
    name=PACKAGE,
    version=_get_version(),
    packages=find_packages(),
    package_data={
        PACKAGE: ['support_data/*', 'support_data/conf/*'],
    },
    author=['Nima Ghorbani'],
    author_email=['nghorbani@tue.mpg.de'],
    maintainer='Nima Ghorbani',
    maintainer_email='nghorbani@tue.mpg.de',
    url='https://github.com/nghorbani/moshpp',
    description='Solving optical marker-based mocap with millimeter accuracy.',
    license='See LICENSE',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=['numpy'],
    ext_modules=[
        Extension(
            "moshpp.scan2mesh.mesh_distance.sample2meshdist",
            sourcefiles,
            language="c++",
            **additional_options)],
    include_dirs=['.'],
    dependency_links=[],
    classifiers=[
        "Intended Audience :: Research",
        "Natural Language :: English",
        "Operating System :: POSIX",
        "Operating System :: POSIX :: BSD",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7", ],
)
