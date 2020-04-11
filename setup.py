from distutils.core import setup
from setuptools import find_packages

setup(
    name='offline-rl',
    version='1.0',
    install_requires=['gym', 
                      'numpy', 
                      'mujoco_py', 
                      'h5py', 
                      'mjrl @ git+git://github.com/aravindr93/mjrl@master#egg=mjrl'],
    packages=find_packages(),
)
