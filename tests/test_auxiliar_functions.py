from swan.cosmo.functions import merge_csv, run_command
from pathlib import Path


def test_run_command():
    """Test command invokation."""
    cmd = "echo $(( 4 * 5 ))"

    rs, _ = run_command(cmd)

    assert rs.split()[0].decode() == '20'


def test_merge_csv(tmp_path):
    """Check the merge of csv files containing the Gammas."""
    path_files = "tests/test_files"
    output = Path(tmp_path) / "merged.csv"

    df = merge_csv(path_files, output)

    assert 'gammas' in df.columns
