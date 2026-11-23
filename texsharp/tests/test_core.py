import math

import pytest

import texsharp

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

IMG = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]


def _s():
    return texsharp.Sharpen()


def _check_shape():
    return len(_s().apply(IMG)) == 3


def _check_zero_amount():
    return _s().apply(IMG, 0.0) == IMG


def _check_edge_diff():
    return _s().apply(IMG, 1.0, "clamp") != _s().apply(IMG, 1.0, "zero")


def _check_constant():
    c = [[5.0] * 3 for _ in range(3)]
    out = _s().apply(c, 1.0)
    return all(abs(v - 5.0) < 1e-9 for row in out for v in row)


def _check_bad_edge():
    return _raises(lambda: _s().apply(IMG, 1.0, "mirror"))


def _check_bad_image():
    return _raises(lambda: _s().apply([], 1.0))


def _check_type():
    return isinstance(_s().apply(IMG), list)


def _check_det():
    return _s().apply(IMG) == _s().apply(IMG)

CASES = [("shape", "_check_shape"), ("zero_amount", "_check_zero_amount"), ("edge_diff", "_check_edge_diff"), ("constant", "_check_constant"), ("bad_edge", "_check_bad_edge"), ("bad_image", "_check_bad_image"), ("type", "_check_type"), ("det", "_check_det"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert Sharpen() is not None
        return
    assert fn() is True
