#!/usr/bin/env python
from setuptools import find_packages, setup


setup(
    name="rnn_prof",
    version='0.1-DEV',
    url="https://github.com/Knewton/edm2016",
    author="knewton",
    author_email="help@knewton.com",
    license="Apache License 2.0",
    packages=find_packages(),
    install_requires=open('requirements.in', 'r').readlines(),
    tests_require=open('requirements.testing.in', 'r').readlines(),
    description="Code for our EDM 2016 submission including DKT and IRT variants",
    entry_points="""
        [console_scripts]
        rnn_prof=rnn_prof.cli:main
    """,
    long_description="\n" + open('README.md', 'r').read()
)
