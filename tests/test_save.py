from dataCAT import Database
from swan.data.save_data import main
import argparse
import os

CSV_FILE = "tests/test_files/thousand.csv"


def test_cosmo_main(mocker):
    """
    Test the call to the main function is cosmo
    """
    # Mock the CLI
    mocker.patch("argparse.ArgumentParser.parse_args", return_value=argparse.Namespace(
        csv=CSV_FILE, uri='localhost', anchor='O1'))

    # Mock the MongoDB interface
    mocker.patch.object(Database, 'update_mongodb')

    main()


def remove_files():
    """
    Remove unused files
    """
    files = ["QD_database.csv", "job_settings.yaml", "ligand_database.csv", "structures.hdf5"]

    for f in files:
        if os.path.exists(f):
            os.remove(f)
