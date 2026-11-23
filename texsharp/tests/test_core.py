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


def _check_kernel_norm():
    """公开核与隐式合成核的权重和均为 1。"""
    flat = [w for row in texsharp.KERNEL for w in row]
    return abs(math.fsum(flat) - 1.0) < 1e-12


def _check_range():
    """锐化不得凭空放大常量区：常量图逐元素精确不变（无振铃）。"""
    big = [[3.25] * 8 for _ in range(8)]
    for amount in (0.5, 1.0, 5.0):
        out = _s().apply(big, amount)
        if any(v != 3.25 for row in out for v in row):
            return False
    return True


def _check_small_image():
    """窄于核宽/高度为 1 的图像也必须合法返回。"""
    for img in ([[5.0]], [[1.0, 2.0]], [[1.0], [2.0]]):
        out = _s().apply(img, 1.0, "zero")
        if len(out) != len(img) or any(len(r) != len(img[0]) for r in out):
            return False
    one = _s().apply([[5.0]], 1.0, "clamp")
    return one == [[5.0]]


def _check_nonfinite():
    return (
        _raises(lambda: _s().apply([[float("nan")]]))
        and _raises(lambda: _s().apply([[1.0, float("inf")]]))
        and _raises(lambda: _s().apply(IMG, float("nan")))
        and _raises(lambda: _s().apply(IMG, float("inf")))
    )


def _check_ragged():
    return _raises(lambda: _s().apply([[1.0, 2.0], [3.0]]))


def _check_amount_type():
    return _raises(lambda: _s().apply(IMG, "1.0"))


def _check_copy():
    """返回新对象，修改输出不得影响输入。"""
    img = [row[:] for row in IMG]
    out = _s().apply(img)
    out[0][0] = -1.0
    return img[0][0] == 1.0


def _check_perf():
    """10 万像素 < 300 ms。"""
    import time

    n = 317  # 100489 像素
    img = [[float((x * 7 + y * 13) % 251) for x in range(n)] for y in range(n)]
    start = time.perf_counter()
    _s().apply(img, 1.0, "clamp")
    return (time.perf_counter() - start) < 0.3


def _check_conc():
    """并发调用与串行结果一致。"""
    import threading

    img = [[float((x * 5 + y * 3) % 37) for x in range(24)] for y in range(24)]
    expected = _s().apply(img, 1.0, "clamp")
    out = []
    lock = threading.Lock()

    def work():
        r = _s().apply(img, 1.0, "clamp")
        with lock:
            out.append(r)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return len(out) == 8 and all(r == expected for r in out)


CASES = [("shape", "_check_shape"), ("zero_amount", "_check_zero_amount"), ("edge_diff", "_check_edge_diff"), ("constant", "_check_constant"), ("bad_edge", "_check_bad_edge"), ("bad_image", "_check_bad_image"), ("type", "_check_type"), ("det", "_check_det"), ("kernel_norm", "_check_kernel_norm"), ("range", "_check_range"), ("small_image", "_check_small_image"), ("nonfinite", "_check_nonfinite"), ("ragged", "_check_ragged"), ("amount_type", "_check_amount_type"), ("copy", "_check_copy"), ("perf", "_check_perf"), ("conc", "_check_conc")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert texsharp.Sharpen() is not None
        return
    assert fn() is True
