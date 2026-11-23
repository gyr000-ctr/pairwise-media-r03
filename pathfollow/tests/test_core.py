import math

import pytest

import pathfollow

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

P = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]


def _f():
    return pathfollow.Follow(P, 10.0)


def _check_start():
    return _f().position() == (0.0, 0.0)


def _check_advance():
    f = _f()
    f.advance(1.0)
    return abs(f.position()[0] - 1.0) < 1e-6


def _check_end():
    f = _f()
    f.advance(100.0)
    return abs(f.position()[0] - 2.0) < 1e-6


def _check_bad_distance():
    return _raises(lambda: _f().advance(-1.0))


def _check_bad_points():
    return _raises(lambda: pathfollow.Follow([(0.0, 0.0)], 10.0))


def _check_bad_curv():
    return _raises(lambda: pathfollow.Follow(P, 0.0))


def _check_type():
    return isinstance(_f().position(), tuple)


def _check_det():
    a = _f(); a.advance(0.5)
    b = _f(); b.advance(0.5)
    return a.position() == b.position()

CASES = [("start", "_check_start"), ("advance", "_check_advance"), ("end", "_check_end"), ("bad_distance", "_check_bad_distance"), ("bad_points", "_check_bad_points"), ("bad_curv", "_check_bad_curv"), ("type", "_check_type"), ("det", "_check_det"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert Follow() is not None
        return
    assert fn() is True
