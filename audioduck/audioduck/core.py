"""audioduck：侧链闪避（sidechain ducking）。

语义
----
``process(samples)`` 逐样本处理侧链信号：

* 非零样本视为“触发”：增益按 ``attack`` 时间常数指数逼近
  ``10 ** (depth_db / 20)``，因此触发期间增益单调不增、且不会越过目标；
* 零样本视为“释放”：增益按 ``release`` 时间常数指数逼近 1.0，足够接近
  时精确吸附到 1.0，恢复后误差为 0（远小于 1e-6）；
* 增益状态保存在实例上、逐样本推进，跨 ``process`` 调用连续，分块处理
  与整段处理逐位一致；
* 返回 ``samples * gain`` 的新列表，不修改输入；空输入返回 ``[]``。

包络形式
--------
逐样本一阶指数平滑：``g += (goal - g) * alpha``，其中

    alpha_attack  = 1 - exp(-1 / (attack  * sample_rate))
    alpha_release = 1 - exp(-1 / (release * sample_rate))

``attack`` / ``release`` 以秒为单位，``sample_rate`` 默认 48 kHz，也可在
构造时显式指定。先在当前样本上推进包络、再作用于该样本。

错误分类
--------
* :class:`InvalidParameterError` —— 构造参数非法（非数值、非有限、越界）；
* :class:`SampleTypeError` —— 样本不是实数；
* :class:`NonFiniteSampleError` —— 样本为 NaN / +Inf / -Inf，带
  ``index`` 与 ``kind`` 属性。

后两者都继承 ``ValueError``；前两者同时继承 ``TypeError``，三类互不混淆。
输入在加锁前整体校验，任何错误都不会改变实例状态（原子性）。

并发
----
``process`` 与 ``gain`` 全程持同一把锁，同一实例可被多线程安全调用。
"""

import math
import threading
from numbers import Real

__all__ = [
    "Duck",
    "DuckError",
    "InvalidParameterError",
    "SampleTypeError",
    "NonFiniteSampleError",
]

#: 恢复到距离 1.0 小于该值时吸附为精确 1.0（误差上界远小于 1e-6）。
_RELEASE_SNAP = 1e-9
#: 触发到距离目标小于该值时吸附为精确目标（消除浮点渐近误差）。
_ATTACK_SNAP = 1e-12


class DuckError(Exception):
    """audioduck 相关错误的基类。"""


class InvalidParameterError(ValueError, TypeError, DuckError):
    """构造参数非法；``param`` / ``reason`` 指出出错的参数与原因。"""

    def __init__(self, param, reason):
        self.param = param
        self.reason = reason
        super().__init__("invalid parameter %r: %s" % (param, reason))


class SampleTypeError(ValueError, TypeError, DuckError):
    """样本类型非法（不是实数）；``index`` / ``kind`` 指出位置与类型。"""

    def __init__(self, index, kind):
        self.index = index
        self.kind = kind
        super().__init__("sample[%d] has non-numeric type %s" % (index, kind))


class NonFiniteSampleError(ValueError, DuckError):
    """样本非有限；``index`` 为位置，``kind`` 为 ``"nan"`` / ``"+inf"`` / ``"-inf"``。"""

    def __init__(self, index, kind):
        self.index = index
        self.kind = kind
        super().__init__("sample[%d] is not finite: %s" % (index, kind))


def _finite_real(name, value):
    """校验为有限实数并返回 float；否则抛 InvalidParameterError。"""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InvalidParameterError(
            name, "must be a real number, got %s" % type(value).__name__
        )
    v = float(value)
    if not math.isfinite(v):
        raise InvalidParameterError(name, "must be finite, got %r" % (v,))
    return v


def _positive_real(name, value):
    v = _finite_real(name, value)
    if v <= 0.0:
        raise InvalidParameterError(name, "must be > 0, got %r" % (v,))
    return v


def _non_finite_kind(value):
    """区分 NaN / +Inf / -Inf 三类非有限值。"""
    if value != value:
        return "nan"
    return "+inf" if value > 0.0 else "-inf"


def _validated_samples(samples):
    """把输入整体校验为 ``list``，不触碰实例状态。

    类型错误抛 :class:`SampleTypeError`，NaN/Inf（含超出 float 范围的整数）
    抛 :class:`NonFiniteSampleError`，二者都带出错下标。热路径先用 C 层的
    ``map`` 做整段有限性检查（100 万样本约 4 ms），只有出错时才逐元素定位
    以给出精确下标。
    """
    if samples is None or isinstance(samples, (str, bytes, bytearray)):
        raise SampleTypeError(0, type(samples).__name__)
    try:
        data = list(samples)
    except TypeError:
        raise SampleTypeError(0, type(samples).__name__) from None

    # 全部是内建 int/float 时走快速通道（bool 的类型为 bool，落到慢通道）。
    if set(map(type, data)) <= {float, int}:
        try:
            if all(map(math.isfinite, data)):
                return data
        except OverflowError:
            pass  # 超出 float 范围的整数：交给慢通道精确定位

    checked = []
    for i, x in enumerate(data):
        if isinstance(x, bool) or not isinstance(x, Real):
            raise SampleTypeError(i, type(x).__name__)
        try:
            value = float(x)
        except (OverflowError, ValueError, TypeError):
            raise NonFiniteSampleError(i, "overflow") from None
        if not math.isfinite(value):
            raise NonFiniteSampleError(i, _non_finite_kind(value))
        checked.append(value)
    return checked


class Duck:
    """侧链闪避器。

    参数
    ----
    depth_db: 触发时相对 0 dB 的衰减深度，须为负有限数（如 ``-6.0``）。
    attack: 触发的指数时间常数（秒），正有限数。
    release: 恢复的指数时间常数（秒），正有限数。
    sample_rate: 采样率（Hz），正有限数，默认 48000。
    """

    def __init__(self, depth_db=-6.0, attack=0.01, release=0.2, sample_rate=48000.0):
        depth_db = _finite_real("depth_db", depth_db)
        if depth_db >= 0.0:
            raise InvalidParameterError("depth_db", "must be < 0 dB")
        self.depth_db = depth_db
        self.attack = _positive_real("attack", attack)
        self.release = _positive_real("release", release)
        self.sample_rate = _positive_real("sample_rate", sample_rate)

        # 线性目标增益：depth_db < 0，故 0 < target < 1。
        self._target = 10.0 ** (depth_db / 20.0)
        # 每样本系数：expm1 在小量处精度更好；极端参数下夹紧到 1。
        ka = -math.expm1(-1.0 / (self.attack * self.sample_rate))
        kr = -math.expm1(-1.0 / (self.release * self.sample_rate))
        self._ka = 1.0 if ka > 1.0 else ka
        self._kr = 1.0 if kr > 1.0 else kr

        self._gain = 1.0
        self._lock = threading.Lock()

    def process(self, samples):
        """处理一段侧链信号，返回与输入等长的新 ``list``。

        非零样本触发（attack），零样本释放（release）；增益跨调用连续。
        任意样本非法时抛出分类错误，实例状态保持不变。
        """
        data = _validated_samples(samples)
        n = len(data)
        if n == 0:
            return []

        with self._lock:
            gain = self._gain
            target = self._target
            ka = self._ka

            # 快速路径 1：增益未动且整段静音 —— 输出全零，状态不变。
            if gain == 1.0 and not any(data):
                return [0.0] * n

            # 快速路径 2：增益已停在目标且整段触发 —— 只用乘法。
            if gain == target and all(data):
                return [x * target for x in data]

            kr = self._kr
            release_snap = _RELEASE_SNAP
            attack_snap = _ATTACK_SNAP
            out = [0.0] * n
            for i, x in enumerate(data):
                if x:
                    if gain != target:
                        gain += (target - gain) * ka
                        if gain - target <= attack_snap:
                            gain = target
                    out[i] = x * gain
                elif gain != 1.0:
                    gain += (1.0 - gain) * kr
                    if 1.0 - gain <= release_snap:
                        gain = 1.0

            self._gain = gain
            return out

    def gain(self):
        """当前（上一样本处理后的）线性增益；初始为 1.0。"""
        with self._lock:
            return self._gain
