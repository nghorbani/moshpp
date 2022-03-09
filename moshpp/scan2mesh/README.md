# Scan2Mesh Package

Given a scan (raw data generated from scanner) and a mesh (like SMPL or other model), the package provides function to
calculate the distance between the corresponding scan vertices and the closest point on the mesh surface.

## Installation

0. sudo apt install libeigen3-dev
1. pip install -r requirements.txt
2. sudo apt install libtbb-dev
3. cd mesh_distance
4. make

## Test

python main.py scan_path smpl_model_file_path