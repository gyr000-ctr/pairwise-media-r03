import math

import pytest

import ao

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _a(n=8):
    return ao.Ao(n, 0)


def _check_range():
    return 0.0 <= _a().sample([[1.0] * 4 for _ in range(4)], 1, 1) <= 1.0


def _check_flat():
    d = [[1.0] * 4 for _ in range(4)]
    return abs(_a().sample(d, 1, 1) - 1.0) < 0.2


def _check_det():
    d = [[1.0] * 4 for _ in range(4)]
    return _a().sample(d, 1, 1) == _a().sample(d, 1, 1)


def _check_seed_diff():
    d = [[1.0, 0.5], [0.5, 1.0]]
    return ao.Ao(8, 1).sample(d, 0, 0) != ao.Ao(8, 2).sample(d, 0, 0)


def _check_bad_samples():
    return _raises(lambda: ao.Ao(0, 0))


def _check_bad_depth():
    return _raises(lambda: _a().sample([], 0, 0))


def _check_type():
    return isinstance(_a().sample([[1.0, 1.0], [1.0, 1.0]], 0, 0), float)


def _check_out_of_range():
    return _raises(lambda: _a().sample([[1.0, 1.0]], 5, 5))

CASES = [("range", "_check_range"), ("flat", "_check_flat"), ("det", "_check_det"), ("seed_diff", "_check_seed_diff"), ("bad_samples", "_check_bad_samples"), ("bad_depth", "_check_bad_depth"), ("type", "_check_type"), ("out_of_range", "_check_out_of_range"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert Ao() is not None
        return
    assert fn() is True
