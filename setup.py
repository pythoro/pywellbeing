# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 15:02:12 2019

@author: Reuben
"""

import setuptools

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="pywellbeing",
    version="0.0.1",
    author="Reuben Rusk",
    author_email="pythoro@mindquip.com",
    description="Computational models of human happiness, wellbeing, and motivation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/pythoro/pywellbeing.git',
    project_urls={
        'Documentation': 'https://github.com/pythoro/pywellbeing.git',
        'Source': 'https://github.com/pythoro/pywellbeing.git',
        'Tracker': 'https://github.com/pythoro/pywellbeing/issues',
    },
    download_url="https://github.com/pythoro/pywellbeing/archive/v0.0.1.zip",
    packages=['pywellbeing'],
    keywords=['WELLBEING', 'HAPPINESS', 'COMPUTATIONAL MODEL', 
              'ADAPTIVE MOTIVATION', 'MOTIVATION'],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering",
    ],
    install_requires=['numpy', 'scipy', 'matplotlib'],
)