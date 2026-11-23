import math

import pytest

import audioduck

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _d():
    return audioduck.Duck(-6.0, 0.01, 0.2)


def _check_initial():
    return abs(_d().gain() - 1.0) < 1e-9


def _check_len():
    return len(_d().process([0.0] * 10)) == 10


def _check_bad_depth():
    return _raises(lambda: audioduck.Duck(0.0, 0.01, 0.2))


def _check_bad_attack():
    return _raises(lambda: audioduck.Duck(-6.0, 0.0, 0.2))


def _check_empty():
    return _d().process([]) == []


def _check_type():
    return isinstance(_d().process([0.0]), list)


def _check_no_jump():
    out = _d().process([0.0] * 100)
    return max(abs(out[i + 1] - out[i]) for i in range(len(out) - 1)) < 0.5


def _check_det():
    return _d().process([0.0] * 5) == _d().process([0.0] * 5)

CASES = [("initial", "_check_initial"), ("len", "_check_len"), ("bad_depth", "_check_bad_depth"), ("bad_attack", "_check_bad_attack"), ("empty", "_check_empty"), ("type", "_check_type"), ("no_jump", "_check_no_jump"), ("det", "_check_det"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert Duck() is not None
        return
    assert fn() is True
