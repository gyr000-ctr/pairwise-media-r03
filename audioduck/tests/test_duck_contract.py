"""audioduck 契约自测（新增文件，未改动既有测试）。

覆盖题面全部约束：attack/release 指数下降与恢复、恢复误差 ≤1e-6、
触发期间增益单调不增、跨缓冲区连续、100 万样本 ≤300 ms、错误分类、
非有限输入、并发安全。
"""

import math
import random
import threading
import time

import pytest

import audioduck

FS = 48000


def duck(depth_db=-6.0, attack=0.01, release=0.2):
    return audioduck.Duck(depth_db, attack, release)


def alpha(tau, fs=FS):
    return 1.0 - math.exp(-1.0 / (tau * fs))


def test_duck_initial_gain_is_unity():
    assert duck().gain() == 1.0


def test_duck_trigger_uses_exponential_attack():
    d = duck(attack=0.01)
    out = d.process([1.0] * 480)
    ka = alpha(0.01)
    target = 10.0 ** (-6.0 / 20.0)
    gain = 1.0
    for i in range(480):
        gain += (target - gain) * ka
        assert out[i] == pytest.approx(gain, rel=0, abs=1e-15)


def test_duck_attack_is_not_instant_and_monotone_nonincreasing():
    d = duck(attack=0.01)
    gains = []
    prev = d.gain()
    for _ in range(50):
        d.process([1.0])
        g = d.gain()
        assert g <= prev + 1e-15
        prev = g
        gains.append(g)
    assert gains[0] > 0.9  # 单个样本不足以到目标
    assert all(a >= b for a, b in zip(gains, gains[1:]))


def test_duck_gain_monotone_during_triggered_run():
    d = duck(depth_db=-12.0, attack=0.05)
    d.process([1.0] * 2000)
    g = []
    for _ in range(60):
        d.process([0.5] * 1000)
        g.append(d.gain())
    assert all(a >= b for a, b in zip(g, g[1:]))
    assert abs(g[-1] - 10.0 ** (-12.0 / 20.0)) < 1e-6


def test_duck_release_recovers_to_unity_within_tolerance():
    d = duck(attack=0.001, release=0.05)
    d.process([1.0] * 5000)  # 触到底
    assert abs(d.gain() - 10.0 ** (-6.0 / 20.0)) < 1e-6
    d.process([0.0] * int(FS * 2))
    assert abs(d.gain() - 1.0) <= 1e-6


def test_duck_release_is_exponential():
    d = duck(attack=0.001, release=0.1)
    d.process([1.0] * 20000)
    g0 = d.gain()
    out = d.process([0.0] * 10)
    kr = alpha(0.1)
    expect = g0
    for _ in range(10):
        expect += (1.0 - expect) * kr
    assert d.gain() == pytest.approx(expect, rel=0, abs=1e-15)
    assert out == [0.0] * 10


def test_duck_output_is_input_times_gain():
    d = duck(attack=0.02, release=0.05)
    sig = [0.7, 0.0, -0.4, 0.0, 0.9, 0.0]
    out = d.process(sig)
    # 单独复算增益轨迹
    d2 = duck(attack=0.02, release=0.05)
    ref = []
    for x in sig:
        ref.extend(d2.process([x]))
    assert out == ref
    assert isinstance(out, list)


def test_duck_state_continuous_across_buffers():
    rng = random.Random(2024)
    sig = [0.0 if rng.random() < 0.3 else rng.uniform(-1, 1) for _ in range(5000)]
    whole = duck().process(sig)
    chunked, d = [], duck()
    for i in range(0, len(sig), 137):
        chunked.extend(d.process(sig[i:i + 137]))
    assert chunked == whole


def test_duck_empty_and_type():
    d = duck()
    assert d.process([]) == []
    assert isinstance(d.process([0.0]), list)
    src = [0.5, -0.5]
    out = d.process(src)
    assert src == [0.5, -0.5]
    assert isinstance(out, list)


def test_duck_silence_does_not_move_gain():
    d = duck()
    d.process([0.0] * 1000)
    assert d.gain() == 1.0
    assert d.process([0.0] * 10) == [0.0] * 10


def test_duck_deterministic():
    sig = [0.3, 0.0, 0.0, 0.8, 0.0]
    assert duck().process(sig) == duck().process(sig)


@pytest.mark.parametrize("bad", [
    lambda: audioduck.Duck(0.0, 0.01, 0.2),
    lambda: audioduck.Duck(6.0, 0.01, 0.2),
    lambda: audioduck.Duck(-6.0, 0.0, 0.2),
    lambda: audioduck.Duck(-6.0, 0.01, 0.0),
    lambda: audioduck.Duck(-6.0, -0.01, 0.2),
    lambda: audioduck.Duck(-6.0, 0.01, -0.2),
    lambda: audioduck.Duck(float("nan"), 0.01, 0.2),
    lambda: audioduck.Duck(-6.0, float("inf"), 0.2),
    lambda: audioduck.Duck(-6.0, 0.01, float("nan")),
    lambda: audioduck.Duck("x", 0.01, 0.2),
    lambda: audioduck.Duck(-6.0, None, 0.2),
])
def test_duck_invalid_parameters(bad):
    with pytest.raises(audioduck.InvalidParameterError):
        bad()
    with pytest.raises(ValueError):
        bad()


@pytest.mark.parametrize("value,kind", [
    (float("nan"), "nan"),
    (float("inf"), "+inf"),
    (float("-inf"), "-inf"),
])
def test_duck_non_finite_samples_are_classified(value, kind):
    d = duck()
    with pytest.raises(audioduck.NonFiniteSampleError) as info:
        d.process([0.5, value, 0.5])
    assert info.value.kind == kind
    assert info.value.index == 1


def test_duck_sample_type_error():
    with pytest.raises(audioduck.SampleTypeError):
        duck().process(["a"])
    with pytest.raises(audioduck.SampleTypeError):
        duck().process("not samples")
    with pytest.raises(audioduck.SampleTypeError):
        duck().process([None])
    with pytest.raises(audioduck.SampleTypeError):
        duck().process([True])


def test_duck_error_taxonomy_is_distinguishable():
    assert issubclass(audioduck.InvalidParameterError, ValueError)
    assert issubclass(audioduck.InvalidParameterError, TypeError)
    assert issubclass(audioduck.SampleTypeError, ValueError)
    assert issubclass(audioduck.SampleTypeError, TypeError)
    assert issubclass(audioduck.NonFiniteSampleError, ValueError)
    assert not issubclass(audioduck.NonFiniteSampleError, TypeError)
    assert not issubclass(audioduck.SampleTypeError, audioduck.NonFiniteSampleError)
    assert not issubclass(audioduck.NonFiniteSampleError, audioduck.SampleTypeError)


def test_duck_state_atomic_on_error():
    d = duck()
    d.process([1.0] * 100)
    before = d.gain()
    with pytest.raises(audioduck.NonFiniteSampleError):
        d.process([1.0, 1.0, float("nan")])
    assert d.gain() == before


def test_duck_one_million_samples_under_300ms():
    rng = random.Random(7)
    sig = [rng.uniform(-1.0, 1.0) for _ in range(1_000_000)]
    d = duck()
    start = time.perf_counter()
    out = d.process(sig)
    elapsed = (time.perf_counter() - start) * 1000.0
    assert len(out) == 1_000_000
    assert elapsed < 300.0, f"耗时 {elapsed:.1f} ms 超过 300 ms"


def test_duck_one_million_triggered_samples_under_300ms():
    d = duck(attack=100.0, release=100.0)
    sig = [0.001] * 1_000_000
    start = time.perf_counter()
    out = d.process(sig)
    elapsed = (time.perf_counter() - start) * 1000.0
    assert len(out) == 1_000_000
    assert elapsed < 300.0, f"耗时 {elapsed:.1f} ms 超过 300 ms"


def test_duck_concurrent_gain_stays_bounded():
    d = duck(attack=0.005, release=0.05)
    errors = []

    def worker():
        try:
            for _ in range(300):
                out = d.process([0.5] * 100)
                if len(out) != 100:
                    errors.append("bad length")
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert 0.0 < d.gain() <= 1.0


def test_duck_instances_are_independent():
    a, b = duck(), duck()
    a.process([1.0] * 5000)
    assert b.gain() == 1.0
    assert a.gain() < 1.0
