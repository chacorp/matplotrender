"""
setup.py
----
This is the main setup file for the hallo face animation project. It defines the package
metadata, required dependencies, and provides the entry point for installing the package.

"""

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
    'version': '0.1.0',
    'description': '',
    'long_description': 'render with matplotlib',
    'author': 'Your Name',
    'author_email': 'you@example.com',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'package_dir':{"": "src"},
    'install_requires': install_requires,
    'python_requires': '>=3.8,<4.0',
}


setup(**setup_kwargs)