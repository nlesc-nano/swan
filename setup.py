#!/usr/bin/env python
import os

from setuptools import setup

HERE = os.path.abspath(os.path.dirname(__file__))

version = {}
with open(os.path.join(HERE, 'swan', '__version__.py')) as f:
    exec(f.read(), version)

with open('README.rst') as readme_file:
    README = readme_file.read()

setup(
    name='swan',
    version=version['__version__'],
    description="",
    long_description=README + '\n\n',
    long_description_content_type='text/markdown',
    author="Felipe Zapata",
    author_email='f.zapata@esciencecenter.nl',
    url='https://github.com/nlesc-nano/swan',

    package_dir={'swan': 'swan'},
    include_package_data=True,
    license="Apache Software License 2.0",
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    install_requires=[
        'lie_learn@git+https://github.com/AMLab-Amsterdam/lie_learn',
        'e3nn@git+https://github.com/e3nn/e3nn@main',
        'equivariant-attention@git+https://github.com/nlesc-nano/se3-transformer-public@dev',
        'mendeleev', 'numpy', 'pandas', 'pyyaml', 'scikit-learn',
        'scipy', 'seaborn', 'schema',
        'torch-geometric'],

    extras_require={
        'test': ['coverage', 'mypy', 'pycodestyle', 'pytest>=3.9', 'pytest-cov',
                 'pytest-mock'],
        'doc': ['sphinx', 'sphinx-autodoc-typehints', 'sphinx_rtd_theme',
                'nbsphinx']
    }
)
