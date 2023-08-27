"""Setups up the PettingZoo module."""

from setuptools import find_packages, setup


def get_description():
    """Gets the description from the readme."""
    with open("README.md") as fh:
        long_description = ""
        header_count = 0
        for line in fh:
            if line.startswith("##"):
                header_count += 1
            if header_count < 2:
                long_description += line
            else:
                break
    return header_count, long_description


def get_version():
    """Gets the d4rl version."""
    path = "d4rl/__init__.py"
    with open(path) as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


version = get_version()
header_count, long_description = get_description()

setup(
    name="D4RL",
    version=version,
    author="Farama Foundation",
    author_email="contact@farama.org",
    description="Datasets for Data Driven Deep Reinforcement Learning.",
    url="https://github.com/Farama-Foundation/D4RL",
    license="Apache",
    license_files=("LICENSE",),
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=["Reinforcement Learning", "Datasets", "RL", "AI"],
    python_requires=">=3.7, <3.11",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "d4rl": [
            "locomotion/assets/*",
            "hand_manipulation_suite/assets/*",
            "hand_manipulation_suite/Adroit/*",
            "hand_manipulation_suite/Adroit/gallery/*",
            "hand_manipulation_suite/Adroit/resources/*",
            "hand_manipulation_suite/Adroit/resources/meshes/*",
            "hand_manipulation_suite/Adroit/resources/textures/*",
        ]
    },
    install_requires=[
        "gym<0.24.0",
        "numpy",
        "mujoco_py",
        "pybullet",
        "h5py",
        "termcolor",  # adept_envs dependency
        "click",  # adept_envs dependency
        "dm_control>=1.0.3",
        "mjrl @ git+https://github.com/aravindr93/mjrl@master#egg=mjrl",
    ],
)
