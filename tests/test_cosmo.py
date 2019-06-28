from swan.cosmo.cosmo import compute_activity_coefficient
from swan.utils import Options
from pathlib import Path
import pandas as pd
import os


def test_cosmo_on_chunks(mocker):

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

    finally:
        os.remove("Gammas_0.csv")
