from swan.cat_interface import run_command


def test_run_command():
    """Test command invokation."""
    cmd = "echo $(( 4 * 5 ))"

    rs, _ = run_command(cmd)

    assert rs.split()[0].decode() == '20'
