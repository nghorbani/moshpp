# MoSh++

This repository contains the official chumpy implementation of mocap body solver used for AMASS:

AMASS: Archive of Motion Capture as Surface Shapes\
Naureen Mahmood, Nima Ghorbani, Nikolaus F. Troje, Gerard Pons-Moll, Michael J. Black\
[Full paper](http://files.is.tue.mpg.de/black/papers/amass.pdf) | 
[Video](https://www.youtube.com/watch?v=cceRrlnTCEs&ab_channel=MichaelBlack) | 
[Project website](https://amass.is.tue.mpg.de/) | 
[Poster](http://files.is.tue.mpg.de/black/papers/amass_iccv_poster.pdf)

## Description

This repository holds the code for MoSh++, introduced in [AMASS](http://amass.is.tue.mpg.de/), ICCV'19.
MoSh++ is the upgraded version of [MoSh](https://ps.is.mpg.de/publications/loper-sigasia-2014), Sig.Asia'2014.
Given a *labeled* marker-based motion capture (mocap) c3d file and the *correspondences* 
of the marker labels to the locations on the body, MoSh can
return model parameters for every frame of the mocap sequence. 
The current MoSh++ code works with the following models:

- [SMPL](https://smpl.is.tue.mpg.de/)
- [SMPL+H](http://mano.is.tue.mpg.de/)
- [SMPL-X](https://smpl-x.is.tue.mpg.de/)
- [MANO](http://mano.is.tue.mpg.de/)
- [Objects](https://grab.is.tue.mpg.de/)
- [SMALL](https://smal.is.tue.mpg.de/)

## Installation


The Current repository requires Python 3.7 and chumpy; a CPU based auto-differentiation package.
This package is assumed to be used along with [SOMA](https://github.com/nghorbani/soma), the mocap auto-labeling package.
Please install MoSh++ inside the conda environment of SOMA.
Clone the moshpp repository, and run the following from the root directory:

```
sudo apt install libtbb-dev

pip install -r requirements.txt

cd src/moshpp/scan2mesh
sudo apt install libeigen3-dev
pip install -r requirements.txt
2. sudo apt install libtbb-dev
cd mesh_distance
make

cd ../../../..
python setup.py install
```

## Tutorials
This repository is a complementary package to [SOMA](https://soma.is.tue.mpg.de/), an automatic mocap solver.
Please refer to the [SOMA repository](https://github.com/nghorbani/soma) for tutorials and use cases.

## Citation

Please cite the following paper if you use this code directly or indirectly in your research/projects:

```
@inproceedings{AMASS:2019,
  title={AMASS: Archive of Motion Capture as Surface Shapes},
  author={Mahmood, Naureen and Ghorbani, Nima and Troje, Nikolaus F. and Pons-Moll, Gerard and Black, Michael J.},
  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
  year={2019},
  month = {Oct},
  url = {https://amass.is.tue.mpg.de},
  month_numeric = {10}
}
```

Please consider citing the initial version of MoSh from Loper et al. Sig. Asia'14:

```
   @article{Loper:SIGASIA:2014,
     title = {{MoSh}: Motion and Shape Capture from Sparse Markers},
     author = {Loper, Matthew M. and Mahmood, Naureen and Black, Michael J.},
     address = {New York, NY, USA},
     publisher = {ACM},
     month = nov,
     number = {6},
     volume = {33},
     pages = {220:1--220:13},
     journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH Asia)},
     url = {http://doi.acm.org/10.1145/2661229.2661273},
     year = {2014},
     doi = {10.1145/2661229.2661273}
   }
```
## License

Software Copyright License for **non-commercial scientific research purposes**. Please read carefully
the [terms and conditions](./LICENSE) and any accompanying documentation before you download and/or
use the MoSh++ data and software, (the "Data & Software"), software, scripts, and animations. 
By downloading and/or using the Data & Software (including downloading, cloning, installing, and any other use of this repository), 
you acknowledge that you have read these terms
and conditions, understand them, and agree to be bound by them. If you do not agree with these terms and conditions, you
must not download and/or use the Data & Software. 
Any infringement of the terms of this agreement will automatically terminate
your rights under this [License](./LICENSE).

The software is compiled using CGAL sources following the license in [CGAL_LICENSE.pdf](CGAL_LICENSE.pdf)

## Contact

The code in this repository is developed by 
[Nima Ghorbani](https://nghorbani.github.io/),
[Naureen Mahmood](https://ps.is.tuebingen.mpg.de/person/nmahmood), and 
[Matthew Loper](https://ps.is.mpg.de/~mloper) 
while at [Max-Planck Institute for Intelligent Systems, Tübingen, Germany](https://is.mpg.de/person/nghorbani).

If you have any questions you can contact us at [amass@tuebingen.mpg.de](mailto:amass@tuebingen.mpg.de).

For commercial licensing, contact [ps-licensing@tue.mpg.de](mailto:ps-licensing@tue.mpg.de)
