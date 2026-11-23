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


# 以下为按题面补充的用例：采样数单调收敛、完全遮挡 AO=0、非有限输入分类
# 报错、抖动可复现、性能上界、并发安全。

def _check_fully_occluded():
    # 中心像素被四周更近的几何完全包住 -> AO 精确为 0。
    d = [[0.0] * 5 for _ in range(5)]
    d[2][2] = 1.0
    return _a().sample(d, 2, 2) == 0.0


def _check_monotone():
    # 相邻采样数的结果严格单调收敛：两者之差随采样数增加严格递减。
    d = [[1.0 + 0.2 * x + 0.1 * y for x in range(8)] for y in range(8)]
    counts = (1, 2, 4, 8, 16, 32)
    values = [ao.Ao(n, 0).sample(d, 4, 4) for n in counts]
    gaps = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]
    return all(gaps[i + 1] < gaps[i] for i in range(len(gaps) - 1))


def _check_seed_reproducible():
    d = [[1.0 + 0.05 * x for x in range(6)] for _ in range(6)]
    return (
        ao.Ao(16, 7).sample(d, 3, 3) == ao.Ao(16, 7).sample(d, 3, 3)
        and ao.Ao(16, 7).sample(d, 3, 3) != ao.Ao(16, 8).sample(d, 3, 3)
    )


def _check_non_finite():
    nan = [[float("nan"), 1.0], [1.0, 1.0]]
    inf = [[float("inf"), 1.0], [1.0, 1.0]]
    return _raises(lambda: _a().sample(nan, 0, 0)) and _raises(
        lambda: _a().sample(inf, 0, 0)
    )


def _check_perf():
    import time

    d = [[1.0 + 0.01 * x + 0.02 * y for x in range(100)] for y in range(100)]
    a = ao.Ao(16, 0)
    start = time.perf_counter()
    for y in range(100):
        for x in range(100):
            a.sample(d, x, y)
    return time.perf_counter() - start <= 0.5


def _check_concurrent():
    import threading

    d = [[1.0 + 0.01 * x + 0.02 * y for x in range(32)] for y in range(32)]
    a = ao.Ao(16, 5)
    expected = [a.sample(d, 16, 16) for _ in range(4)]
    results = []

    def work():
        results.append([a.sample(d, 16, 16) for _ in range(4)])

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return all(r == expected for r in results)

CASES = [("range", "_check_range"), ("flat", "_check_flat"), ("det", "_check_det"), ("seed_diff", "_check_seed_diff"), ("bad_samples", "_check_bad_samples"), ("bad_depth", "_check_bad_depth"), ("type", "_check_type"), ("out_of_range", "_check_out_of_range"), ("fully_occluded", "_check_fully_occluded"), ("monotone", "_check_monotone"), ("seed_reproducible", "_check_seed_reproducible"), ("non_finite", "_check_non_finite"), ("perf", "_check_perf"), ("concurrent", "_check_concurrent")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_ao_spec(name, check):
    fn = globals().get(check)
    assert fn is not None, "missing check function %s" % check
    assert fn() is True
