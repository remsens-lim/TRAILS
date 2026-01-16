#!/usr/bin/env python

from setuptools import setup

setup(
    name='TRAILS',
    version='1.0',
    description='TRAjectory-based Identification of Lofted Smoke',
    author='Johanna Roschke',
    author_email='johanna.roschke@uni-leipzig.de',
    url='https://git@github.com:remsens-lim/TRAILS.git',
    license='MIT',
    packages=['trails'],
    package_dir={"": "src"},
    package_data={"": ["*.json"]},
    include_package_data=True,
    install_requires=[
        'numpy',
        'xarray',
        'scipy',
        'scikit-image',
        'opencv-python',
        'matplotlib',
        'pandas',
        'openpyxl',
        'toml',
        'tqdm',
        'pyproj',
        'rasterio',
        'affine',
        'netCDF4',
        'h5netcdf',
        'pathlib'
    ],
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3.9',
    ],
)