from swan.cosmo.functions import (run_command, chunks_of)


def test_run_command():
    """
    Test command invokation
    """
    cmd = "echo $(( 4 * 5 ))"

    rs, err = run_command(cmd)

    assert rs.split()[0].decode() == '20'


def test_chunks():
    """
    Test chunks splitting
    """
    xs = list(range(20))
    s = [sum(x) for x in chunks_of(xs, 5)]
    assert s == [10, 35, 60, 85]
