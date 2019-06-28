from swan.cosmo.cosmo import (call_unifac, compute_activity_coefficient)
from swan.utils import Options
from pathlib import Path
import numpy as np
import pandas as pd
import os


def test_activity_coefficients(mocker):

    output = "Gammas_0.csv"
    try:
        # empty dataframe
        df = pd.DataFrame(columns=["E_solv", "gammas"])

        # Function to mock in the cosmo module
        mocker.patch("swan.cosmo.cosmo.call_mopac", return_value=42)

        # Options to compute the activity coefficient
        d = {"file_smiles": "tests/test_files/Carboxylic_Acids_GDB13.txt",
             "solvent": "CC1=CC=CC=C1",
             "workdir": Path("."), "size_chunk": 1000, "processes": 1,
             "data": df}

        opts = Options(d)

        compute_activity_coefficient(opts)
        assert os.path.exists(output)

    finally:
        os.remove(output)


def test_unifac(mocker):
    """
    Test the call to unifac
    """
    unifac_output = "tests/test_files/unifac_output.out"
    with open(unifac_output, 'br') as f:
        xs = f.read()

    # Mock the call to Unifac
    mocker.patch.dict(os.environ, {'ADFBIN': "tests/test_files/unifac"})
    mocker.patch("swan.cosmo.cosmo.run_command", return_value=(xs, ()))

    opts = {"solvent": "CC1=CC=CC=C1"}

    x = call_unifac(opts, "CO")

    assert np.allclose(x, 13.6296)
