from swan.cosmo.cat_interface import (call_mopac, call_cat_mopac)
from swan.cosmo.cosmo import (call_unifac, compute_activity_coefficient, main)
from swan.utils import Options
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import os
import shutil

smiles_file = "tests/test_files/Carboxylic_Acids_GDB13.txt"


def test_cosmo_main(mocker):
    """
    Test the call to the main function is cosmo
    """

    try:
        # Mock the CLI
        mocker.patch("argparse.ArgumentParser.parse_args", return_value=argparse.Namespace(
            i=smiles_file, csv=None, s="CC1=CC=CC=C1", n=1000, p=1, w="."))

        # Mock the computation of the coefficient
        mocker.patch(
            "swan.cosmo.cosmo.compute_activity_coefficient", return_value=None)

        main()
    finally:
        shutil.rmtree("plams_workdir")


def test_activity_coefficients(mocker):
    """
    Test the activity coefficient calculation
    """
    def call_fun(inp):
        compute_activity_coefficient(opts)
        assert os.path.exists(output)
        os.remove(output)

    output = "Gammas_0.csv"
    # empty dataframe
    df = pd.DataFrame(columns=["E_solv", "gammas"])

    # Function to mock in the cosmo module
    mocker.patch("swan.cosmo.cosmo.call_mopac", return_value=42)

    # Options to compute the activity coefficient
    d = {"file_smiles": smiles_file,
         "solvent": "CC1=CC=CC=C1",
         "workdir": Path("."), "size_chunk": 1000, "processes": 1,
         "data": df}

    opts = Options(d)

    # Check the default case
    call_fun(opts)

    # Check the case when there is already some data
    opts.data["gammas"] = 42.0

    call_fun(opts)


def test_unifac(mocker):
    """
    Test the call to unifac
    """
    unifac_output = "tests/test_files/unifac_output.out"
    with open(unifac_output, 'br') as f:
        xs = f.read()

    # Mock the call to Unifac
    mocker.patch.dict(os.environ, {'ADFBIN': "tests/test_files"})
    mocker.patch("swan.cosmo.cosmo.run_command", return_value=(xs, ()))

    opts = {"solvent": "CC1=CC=CC=C1"}

    x = call_unifac(opts, "CO")

    assert np.allclose(x, 13.6296)

    # Test Failure
    mocker.patch("swan.cosmo.cosmo.run_command", return_value=(b"", ()))
    x = call_unifac(opts, "Wrong_smile")

    assert np.isnan(x)


def test_mopac(mocker):
    """
    Test mock call to ADF/MOPAC
    """
    def side_effect(*args):
        raise ValueError

    # Mock fast sigma call
    mocker.patch.dict(os.environ, {'ADFBIN': "tests/test_files"})
    mocker.patch("swan.cosmo.cat_interface.run_command", return_value=((), ()))
    mocker.patch("swan.cosmo.cat_interface.call_cat_mopac", return_value=(42, 42))
    rs = call_mopac("CO")
    assert rs == (42, 42)

    # Check failure
    mocker.patch("swan.cosmo.cat_interface.call_cat_mopac", side_effect=side_effect)
    rs = call_mopac("Wrong_smile")
    assert rs == (np.nan, np.nan)


def test_call_cat_mopac(mocker, tmp_path):
    """
    Check the call to `get_solv`
    """
    answer = ([42], ())
    mocker.patch("swan.cosmo.cat_interface.get_solv", return_value=answer)

    rs = call_cat_mopac(Path(tmp_path), "CO", ["Toluene.coskf"])

    assert rs == (42, np.nan)
