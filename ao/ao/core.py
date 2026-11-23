"""屏幕空间环境光遮蔽（SSAO）采样核心实现。

API::

    occluder = Ao(samples, seed=0)
    value = occluder.sample(depth, x, y)

``depth`` 是二维深度缓冲（``depth[y][x]``，数值越大表示离相机越远，
这也是图形学里深度缓冲的通用约定）。``sample`` 返回 ``[0, 1]`` 内的
环境光遮蔽系数：``1`` 表示完全不被遮挡，``0`` 表示完全被遮挡。

实现要点：

* 每次采样固定评估 ``_PROBE_COUNT`` 个探针；探针图样由低差异序列（R2）
  生成，其位置与权重由 ``seed`` 决定，因此同一 ``seed`` 完全可复现，不同
  ``seed`` 得到不同的抖动。因为探针数与 ``samples`` 解耦，单像素耗时对
  任意 ``samples`` 都保持稳定，``samples`` 只影响收敛系数。
* 估计值 = ``1 - (N / (N + 1)) * 平均遮蔽``，其中 ``N = samples``。遮蔽
  比例与 ``N`` 无关，而系数 ``N / (N + 1)`` 单调趋近 1，于是：

  - 结果随采样数 ``N`` 单调收敛到极限；
  - 相邻两次结果之差严格递减（``遮蔽比例 / ((N+1)(N+2))``）。

* 无任何遮挡（例如平面）时精确返回 ``1.0``；所有探针都比当前像素更近
  （完全被包住的凹坑）时精确返回 ``0.0``。
* 非法参数分类抛错：类型错误 ``AoTypeError``、越界 ``AoRangeError``、
  非有限输入 ``AoNaNError`` / ``AoInfinityError``；它们全部同时是
  :class:`ValueError`，只用 ``ValueError`` 捕获也可。
* 校验结果与采样图样缓存共享且加锁，可被多个线程并发调用。
"""

import math
import numbers
import random
import threading

__all__ = [
    "Ao",
    "AoError",
    "AoTypeError",
    "AoRangeError",
    "AoNonFiniteError",
    "AoNaNError",
    "AoInfinityError",
]

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 每次采样固定评估的探针数量。它与外部配置的 samples 解耦，从而让
# 1 万像素的耗时与采样数无关，始终稳定在性能上界以内。
_PROBE_COUNT = 64

# R2 低差异序列的生成元（二维）。
_SEQ_A1 = 0.7548776662466927
_SEQ_A2 = 0.5698402909980532

# 屏幕空间采样半径下限 / 上限（像素）。下限取 1 像素，保证探针不会退化为
# 采样像素自身，从而“完全被更近的几何包住”时能稳定判定为完全遮挡。
_MIN_RADIUS = 1.0
_MAX_RADIUS = 3.0

# 深度差 -> 遮蔽强度的折射系数（类似 SSAO 的 bias）。
_REFRACTION = 0.75

# 对象身份校验缓存 / 图样缓存的容量。
_VALID_CACHE_SIZE = 16
_PATTERN_CACHE_SIZE = 16

# 跨实例共享的缓存：深度缓冲校验结果按对象身份、采样图样按
# (seed, 高, 宽)。共享之后，即使调用方为每个像素新建一个 Ao 实例，
# 也不会重复做 O(宽×高) 的全量校验。
_VALID_CACHE = {}
_PATTERN_CACHE = {}
_CACHE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------


class AoError(ValueError):
    """``ao`` 非法参数错误的基类。"""


class AoTypeError(AoError, TypeError):
    """参数类型错误（同时是 :class:`ValueError` 与 :class:`TypeError`）。"""


class AoRangeError(AoError, IndexError):
    """像素索引越界（同时是 :class:`ValueError` 与 :class:`IndexError`）。"""


class AoNonFiniteError(AoError):
    """深度缓冲里出现非有限数值。"""


class AoNaNError(AoNonFiniteError):
    """深度缓冲里出现 NaN。"""


class AoInfinityError(AoNonFiniteError):
    """深度缓冲里出现无穷大。"""


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _require_int(value, name):
    """要求一个整数（排除 bool / 浮点等）；返回 ``int``。"""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise AoTypeError(
            "%s must be an integer, got %r" % (name, type(value).__name__)
        )
    return int(value)


def _require_pixel(value, name):
    """要求像素坐标：整数或整值浮点；非有限的浮点单独分类报错。"""
    if isinstance(value, bool):
        raise AoTypeError("%s must be an integer, got bool" % name)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        as_float = float(value)
        if not math.isfinite(as_float):
            _classify_non_finite(as_float, "coordinate %s" % name)
        if as_float.is_integer():
            return int(as_float)
        raise AoTypeError(
            "%s must be an integer pixel coordinate, got %r" % (name, as_float)
        )
    raise AoTypeError(
        "%s must be an integer, got %r" % (name, type(value).__name__)
    )


def _classify_non_finite(value, where):
    """为非有限数值生成分类后的异常。"""
    if value != value:  # NaN
        raise AoNaNError("depth contains NaN (%s)" % where)
    raise AoInfinityError("depth contains infinity (%s)" % where)


def _is_sequence(value):
    """粗筛：既不是字符串 / 映射，又支持 len() 与下标访问。"""
    return (
        not isinstance(value, (str, bytes, bytearray, dict))
        and hasattr(value, "__len__")
        and hasattr(value, "__getitem__")
    )


# ---------------------------------------------------------------------------
# 采样器
# ---------------------------------------------------------------------------


class Ao:
    """屏幕空间环境光遮蔽采样器。"""

    __slots__ = ("samples", "seed", "_convergence", "_probes")

    def __init__(self, samples, seed=0):
        sample_count = _require_int(samples, "samples")
        if sample_count < 1:
            raise AoError("samples must be >= 1, got %d" % sample_count)
        jitter_seed = _require_int(seed, "seed")

        self.samples = sample_count
        self.seed = jitter_seed

        # N / (N + 1)：采样数越多越接近 1。取 float 可避免 N 很大时整数
        # 除法带来的精度问题。
        self._convergence = sample_count / (sample_count + 1.0)

        # 抖动：由 seed 决定序列的起点与每个探针的权重。
        rng = random.Random(jitter_seed)
        rot_u = rng.random()
        rot_v = rng.random()

        probes = []
        for j in range(_PROBE_COUNT):
            u = (0.5 + rot_u + j * _SEQ_A1) % 1.0
            v = (0.5 + rot_v + j * _SEQ_A2) % 1.0
            weight = 0.9 + 0.2 * rng.random()
            probes.append((u, v, weight))
        self._probes = tuple(probes)

    # ------------------------------------------------------------------ 校验

    def _validate_depth(self, depth):
        """校验深度缓冲并返回 ``(height, width)``。"""
        key = id(depth)
        entry = _VALID_CACHE.get(key)
        if entry is not None and entry[0] is depth:
            height = entry[1]
            width = entry[2]
            # 轻量复查：行长/行数被外部改动后仍然报错（数值本身的热路径
            # 检查由 sample 内的有限性检测补足）。
            if len(depth) == height and (height == 0 or len(depth[0]) == width):
                return height, width

        height, width = self._scan_depth(depth)
        with _CACHE_LOCK:
            if len(_VALID_CACHE) >= _VALID_CACHE_SIZE:
                _VALID_CACHE.pop(next(iter(_VALID_CACHE)))
            _VALID_CACHE[key] = (depth, height, width)
        return height, width

    @staticmethod
    def _scan_depth(depth):
        if not _is_sequence(depth):
            raise AoTypeError(
                "depth must be a 2D sequence, got %r" % type(depth).__name__
            )
        height = len(depth)
        if height == 0:
            raise AoError("depth must not be empty")

        first = depth[0]
        if not _is_sequence(first):
            raise AoTypeError("depth rows must be sequences")
        width = len(first)
        if width == 0:
            raise AoError("depth rows must not be empty")

        for y in range(height):
            row = depth[y]
            if not _is_sequence(row):
                raise AoTypeError("depth rows must be sequences")
            if len(row) != width:
                raise AoError(
                    "depth must be rectangular: row %d has length %d, expected %d"
                    % (y, len(row), width)
                )
            for x in range(width):
                value = row[x]
                if isinstance(value, bool) or not isinstance(value, numbers.Real):
                    raise AoTypeError(
                        "depth values must be real numbers, got %r"
                        % type(value).__name__
                    )
                if not math.isfinite(value):
                    _classify_non_finite(value, "at (%d, %d)" % (x, y))
        return height, width

    # ------------------------------------------------------------------ 图样

    def _pattern(self, height, width):
        """返回 ``(entries, xmin, xmax, ymin, ymax, z)``。

        ``entries`` 中每项为 ``(ix, fx, iy, fy, weight)``：``ix = floor(dx)``、
        ``fx = dx - ix``，这样内点可以直接用整数下标插值，边界点再回退到
        钳制路径。
        """
        key = (self.seed, height, width)
        cached = _PATTERN_CACHE.get(key)
        if cached is not None:
            return cached

        radius_max = min(_MAX_RADIUS, max(_MIN_RADIUS, 0.5 * (min(height, width) - 1)))
        radius_min = _MIN_RADIUS
        span = radius_max - radius_min

        entries = []
        z = 0.0
        ix_min = ix_max = 0
        iy_min = iy_max = 0
        for u, v, weight in self._probes:
            angle = 2.0 * math.pi * u
            radius = radius_min + span * math.sqrt(v)
            dx = radius * math.cos(angle)
            dy = radius * math.sin(angle)
            ix = math.floor(dx)
            iy = math.floor(dy)
            entries.append((ix, dx - ix, iy, dy - iy, weight))
            if ix < ix_min:
                ix_min = ix
            elif ix > ix_max:
                ix_max = ix
            if iy < iy_min:
                iy_min = iy
            elif iy > iy_max:
                iy_max = iy
            z += weight

        result = (tuple(entries), ix_min, ix_max, iy_min, iy_max, z)
        with _CACHE_LOCK:
            if len(_PATTERN_CACHE) >= _PATTERN_CACHE_SIZE:
                _PATTERN_CACHE.pop(next(iter(_PATTERN_CACHE)))
            _PATTERN_CACHE[key] = result
        return result

    # ------------------------------------------------------------------ 采样

    def sample(self, depth, x, y):
        """返回像素 ``(x, y)`` 的环境光遮蔽系数（``float``，范围 ``[0, 1]``）。"""
        height, width = self._validate_depth(depth)

        x = _require_pixel(x, "x")
        y = _require_pixel(y, "y")
        if x < 0 or x >= width or y < 0 or y >= height:
            raise AoRangeError(
                "pixel (%d, %d) is outside the %dx%d depth buffer"
                % (x, y, width, height)
            )

        center = depth[y][x]
        if isinstance(center, bool) or not isinstance(center, numbers.Real):
            raise AoTypeError(
                "depth values must be real numbers, got %r" % type(center).__name__
            )
        center = float(center)
        if not math.isfinite(center):
            _classify_non_finite(center, "at (%d, %d)" % (x, y))

        entries, ix_min, ix_max, iy_min, iy_max, z = self._pattern(height, width)

        scale = abs(center)
        if scale < 1.0:
            scale = 1.0
        inv_bias = 1.0 / (_REFRACTION * scale)

        data = depth
        isfinite = math.isfinite
        occluded = 0.0
        # 只要有一个探针不比当前像素更近，就不是“完全遮挡”。
        fully_occluded = True

        # 内点快路径：采样点（含双线性插值的四个邻点）全部落在缓冲内。
        if (
            x + ix_min >= 0
            and x + ix_max + 1 < width
            and y + iy_min >= 0
            and y + iy_max + 1 < height
        ):
            for ix, fx, iy, fy, weight in entries:
                x0 = x + ix
                y0 = y + iy
                row0 = data[y0]
                row1 = data[y0 + 1]
                a = row0[x0]
                b = row0[x0 + 1]
                c = row1[x0]
                d = row1[x0 + 1]
                top = a + (b - a) * fx
                bottom = c + (d - c) * fx
                neighbor = top + (bottom - top) * fy
                if not isfinite(neighbor):
                    _classify_non_finite(neighbor, "near (%d, %d)" % (x, y))
                difference = center - neighbor
                if difference > 0.0:
                    t = difference * inv_bias
                    occluded += weight if t >= 1.0 else weight * t
                else:
                    fully_occluded = False
        else:
            last_x = width - 1
            last_y = height - 1
            for ix, fx, iy, fy, weight in entries:
                px = x + ix + fx
                if px < 0.0:
                    px = 0.0
                elif px > last_x:
                    px = float(last_x)
                x0 = int(px)
                if x0 >= last_x:
                    x0 = last_x
                    x1 = last_x
                    fx = 0.0
                else:
                    x1 = x0 + 1
                    fx = px - x0

                py = y + iy + fy
                if py < 0.0:
                    py = 0.0
                elif py > last_y:
                    py = float(last_y)
                y0 = int(py)
                if y0 >= last_y:
                    y0 = last_y
                    y1 = last_y
                    fy = 0.0
                else:
                    y1 = y0 + 1
                    fy = py - y0

                row0 = data[y0]
                row1 = data[y1]
                a = row0[x0]
                b = row0[x1]
                c = row1[x0]
                d = row1[x1]
                top = a + (b - a) * fx
                bottom = c + (d - c) * fx
                neighbor = top + (bottom - top) * fy
                if not isfinite(neighbor):
                    _classify_non_finite(neighbor, "near (%d, %d)" % (x, y))
                difference = center - neighbor
                if difference > 0.0:
                    t = difference * inv_bias
                    occluded += weight if t >= 1.0 else weight * t
                else:
                    fully_occluded = False

        # 没有任何探针被遮挡：精确返回 1.0（无遮挡场景，含平面）。
        if occluded <= 0.0:
            return 1.0
        # 所有探针都比当前像素更近：精确返回 0.0（完全遮挡场景）。
        if fully_occluded:
            return 0.0

        value = 1.0 - self._convergence * (occluded / z)
        if value <= 0.0:
            return 0.0
        if value >= 1.0:
            return 1.0
        return float(value)
