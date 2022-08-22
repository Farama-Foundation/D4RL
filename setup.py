from distutils.core import setup
from platform import platform

from setuptools import find_packages

setup(
    name='d4rl',
    version='1.1',
    install_requires=[
        'gym==0.20.0',
        'psutil',
        'numpy',
        'mujoco_py',
        'pybullet',
        'h5py',
        'termcolor',  # adept_envs dependency
        'click',  # adept_envs dependency
        'dm_control>=1.0.5',
        'gfootball==2.10.2',
        'patchelf==0.14.5.0',
        'six==1.16.0',
        "progressbar==2.5"
    ],
    packages=find_packages(),
    package_data={'d4rl': ['locomotion/assets/*',
                           'hand_manipulation_suite/assets/*',
                           'hand_manipulation_suite/Adroit/*',
                           'hand_manipulation_suite/Adroit/gallery/*',
                           'hand_manipulation_suite/Adroit/resources/*',
                           'hand_manipulation_suite/Adroit/resources/meshes/*',
                           'hand_manipulation_suite/Adroit/resources/textures/*',
                           ]},
    include_package_data=True,
)
