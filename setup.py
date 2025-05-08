# -*- coding: utf-8 -*-
from setuptools import setup

install_requires = [
 'trimesh',
 'numpy',
 'matplotlib',
 'tqdm'
]

setup_kwargs = {
    'name': 'matplotrender',
    'version': '0.1.1',
    'description': 'render 3d mesh using matplotlib',
    'long_description': 'render image and video with 3d mesh using matplotlib',
    'author': 'Sihun Cha',
    'author_email': 'sihun.cha@gmail.com',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'https://github.com/chacorp/matplotrender',
    'package_dir':{"": "src"},
    'install_requires': install_requires,
    'python_requires': '>=3.8,<4.0',
    'classifiers': [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
}


setup(**setup_kwargs)