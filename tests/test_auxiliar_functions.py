from swan.cosmo.functions import run_command


def test_run_command():
    """
    Test helper functions
    """
    cmd = "echo $(( 4 * 5 ))"

    rs, err = run_command(cmd)

    assert rs.split()[0].decode() == '20'
