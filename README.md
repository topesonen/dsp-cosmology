# dsp-cosmology
Priors for the unknown properties of the Local Group from large cosmological simulations

The Local Group is the group of galaxies that contains the Milky Way, M31, and around 100 smaller galaxies. Some of its properties are known to great precision, but others, including the masses of the galaxies and the amount of intergalactic matter, the group's total energy and angular momentum, and its formation history, are difficult to measure directly. One way to infer or constrain these is to consider Local Group analogues from large cosmological simulations. In the project, the students will be provided access to data from a very large cosmological simulation (Illustris-TNG-300, https://www.tng-project.org/data/downloads/TNG300-1/), identify analogues of the Local Group, and can use their choice of statistical or machine-learning methods to derive conditional probability distributions for their properties, given different sets of observational constraints.

Data: https://www.tng-project.org/data/downloads/TNG300-1/


TNG300 + illustris_python (quick setup notes)
================================================

Folder layout used
------------------
~/Documents/uni/dsp/
  .venv/                      Python venv (one for the workspace)
  illustris_python/           helper library (git clone)
  tng300/outputs/             data root used as basePath
    groups_099/               downloaded group catalog parts (TNG z=0)

1) Create and use a venv (optional, depending on your preference / environment)
-----------------------
cd ~/Documents/uni/dsp
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel

If python3 -m venv fails (Ubuntu/Debian):
sudo apt update
sudo apt install -y python3-venv

2) Install illustris_python into the venv
----------------------------------------
cd ~/Documents/uni/dsp
git clone https://github.com/illustristng/illustris_python.git
python -m pip install -e ./illustris_python

Install common deps:
python -m pip install numpy h5py matplotlib

3) Download TNG300 group catalog (snapshot 99, z=0)
--------------------------------------------------
Create the folder as groups_099:
mkdir -p ~/Documents/uni/dsp/tng300/outputs/groups_099
cd ~/Documents/uni/dsp/tng300/outputs/groups_099

Download using the TNG API:
wget -nd -nc -nv -e robots=off -l 1 -r -A hdf5 --content-disposition --header="API-Key: 7a37e88e2f9bfb70e539a126ef60e57a" "http://www.tng-project.org/api/TNG300-1/files/groupcat-99/?format=api" 


You should get many files like:
fof_subhalo_tab_099.0.hdf5
fof_subhalo_tab_099.1.hdf5
...

4) Load subhalos in Python (TNG: snapNum=99)
-------------------------------------------
import os
import illustris_python as il

basePath = os.path.expanduser("~/Documents/uni/dsp/tng300/outputs")
fields = ["SubhaloMass", "SubhaloSFRinRad"]

subhalos = il.groupcat.loadSubhalos(basePath, 99, fields=fields)
subhalos["count"], subhalos["SubhaloMass"].shape

