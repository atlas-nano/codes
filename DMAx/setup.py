from setuptools import setup, find_packages

setup(
    name='DMAx',
    version='1.0',
    author='Ricardo Silva Buarque',
    author_email='rsilvabuarque@ucsd.edu',
    description='A high-throughput workflow for Dynamic Mechanical Analysis simulations with LAMMPS',
    package_dir={"": "DMAx"},
    packages=find_packages(where="DMAx"),
    url="https://github.com/rsilvabuarque/DMAx",
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'scikit-learn',
        'mastercurves'
    ],
)
