from swan.cosmo.functions import (chunks_of, merge_csv, run_command)
from pathlib import Path


def test_run_command():
    """
    Test command invokation
    """
    cmd = "echo $(( 4 * 5 ))"

    rs, _ = run_command(cmd)

    assert rs.split()[0].decode() == '20'


def test_chunks():
    """
    Test chunks splitting
    """
    xs = list(range(20))
    s = [sum(x) for x in chunks_of(xs, 5)]
    assert s == [10, 35, 60, 85]


def test_merge_csv(tmp_path):
    """
    Check the merge of csv files containing the Gammas
    """
    path_files = "tests/test_files"
    output = Path(tmp_path) / "merged.csv"

    df = merge_csv(path_files, output)

    assert 'gammas' in df.columns
