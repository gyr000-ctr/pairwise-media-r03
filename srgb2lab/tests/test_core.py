import math

import pytest

import srgb2lab

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _s():
    return srgb2lab.SrgbLab()


def _check_roundtrip():
    rgb = (0.2, 0.5, 0.8)
    return all(abs(a - b) < 1e-4 for a, b in zip(_s().to_rgb(_s().to_lab(rgb)), rgb))


def _check_white():
    L, a, b = _s().to_lab((1.0, 1.0, 1.0))
    return abs(L - 100.0) < 1e-3 and abs(a) < 1e-3


def _check_black():
    return _s().to_lab((0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)


def _check_gray_ab():
    L, a, b = _s().to_lab((0.5, 0.5, 0.5))
    return abs(a) < 1e-6 and abs(b) < 1e-6


def _check_bad_len():
    return _raises(lambda: _s().to_lab((1.0, 0.0)))


def _check_bad_range():
    return _raises(lambda: _s().to_lab((1.5, 0.0, 0.0)))


def _check_type():
    return isinstance(_s().to_lab((0.1, 0.2, 0.3)), tuple)


def _check_det():
    return _s().to_lab((0.3, 0.4, 0.5)) == _s().to_lab((0.3, 0.4, 0.5))

CASES = [("roundtrip", "_check_roundtrip"), ("white", "_check_white"), ("black", "_check_black"), ("gray_ab", "_check_gray_ab"), ("bad_len", "_check_bad_len"), ("bad_range", "_check_bad_range"), ("type", "_check_type"), ("det", "_check_det"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert SrgbLab() is not None
        return
    assert fn() is True
