import sys

from psbody.mesh import Mesh
from psbody.smpl.serialization import load_model

from moshpp.scan2mesh.mesh_distance_main import ScanToMesh
from moshpp.scan2mesh.mesh_distance_main import sample_from_mesh
from moshpp.scan2mesh.robustifiers import GMOf

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception("Please provie the scan_path and smpl model path")
    scan_path = sys.argv[1]
    model_path = sys.argv[2]
    scan = Mesh(filename=scan_path)
    scan_sampler = sample_from_mesh(scan, sample_type='uniformly-from-vertices')
    model = load_model(model_path)
    s2m_dist = ScanToMesh(scan, model.r, model.f,
                          scan_sampler=scan_sampler, signed=False,
                          rho=lambda x: GMOf(x, sigma=0.1))
    print(s2m_dist)
