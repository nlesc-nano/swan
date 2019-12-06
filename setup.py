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
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    entry_points={
        'console_scripts': [
            'cosmo=swan.cosmo.cosmo:main',
            'modeller=swan.models.models:main',
            'save_dataset_in_db=swan.data.save_data:main'
        ]
    },
    test_suite='tests',
    install_requires=[
        'CAT@git+https://github.com/nlesc-nano/CAT@master',
        'nano-CAT@git+https://github.com/nlesc-nano/nano-CAT@master',
        'data-CAT@git+https://github.com/nlesc-nano/data-CAT@master',
        'numpy', 'pandas', 'pyyaml>=5.1.1', 'seaborn', 'schema', 'sqlalchemy'],
    setup_requires=[
        # dependency for `python setup.py test`
        'pytest-runner',
        # dependencies for `python setup.py build_sphinx`
        'sphinx',
        'sphinx_rtd_theme',
        'recommonmark'
    ],
    tests_require=[
        'pytest',
        'pytest-cov',
        'pycodestyle',
    ],
    extras_require={
        'test': ['coverage', 'pycodestyle', 'pytest>=3.9', 'pytest-cov', 'pytest-mock'],
        'doc': ['sphinx', 'sphinx-autodoc-typehints', 'sphinx_rtd_theme', 'nbsphinx']
    }
)
