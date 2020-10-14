
.. image:: https://api.codacy.com/project/badge/Grade/e410d9da7b654d2caf67481f33ae2de7
    :target: https://www.codacy.com/app/nlesc-jcer/swan?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=nlesc-nano/swan&amp;utm_campaign=Badge_Grade
.. image:: https://readthedocs.org/projects/swan/badge/?version=latest
   :target: https://swan.readthedocs.io/en/latest/?badge=latest
.. image:: https://github.com/nlesc-nano/swan/workflows/build%20with%20conda/badge.svg
   :target: https://github.com/nlesc-nano/swan/actions

#####################################
Screening Workflows And Nanomaterials
#####################################

🦢 **Swan** is a Python pacakge to create statistical models to predict molecular properties. See Documentation_.


🛠 Installation
===============

- Download miniconda for python3: miniconda_ (also you can install the complete anaconda_ version).

- Install according to: installConda_.

- Create a new virtual environment using the following commands:

  - ``conda create -n swan``

- Activate the new virtual environment

  - ``conda activate swan``

To exit the virtual environment type  ``conda deactivate``.


.. _dependecies:

Dependencies installation
-------------------------

- Type in your terminal:

  ``conda activate swan``

Using the conda environment the following packages should be installed:


- install OpenMPI_, RDKit_ and H5PY_:

  - `conda install -y -q -c conda-forge openmpi h5py rdkit`

- install Pytorch_ according to this_ recipe

- install `Pytorch_Geometric dependencies <https://github.com/rusty1s/pytorch_geometric#installation>`


.. _installation:

Package installation
--------------------
Finally install the package:

- Install **swan** using pip:
  - ``pip install git+https://github.com/nlesc-nano/swan.git@master``

Now you are ready to use *swan*.


  **Notes:**

  - Once the libraries and the virtual environment are installed, you only need to type
    ``conda activate swan`` each time that you want to use the software.

.. _Documentation: https://swan.readthedocs.io/en/latest/
.. _miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _anaconda: https://www.anaconda.com/distribution/#download-section
.. _installConda: https://conda.io/projects/conda/en/latest/user-guide/install/index.html
.. _Pytorch: https://pytorch.org
.. _RDKit: https://www.rdkit.org
.. _H5PY: https://www.h5py.org/
.. _this: https://pytorch.org/get-started/locally/
.. _OpenMPI: https://www.open-mpi.org/
