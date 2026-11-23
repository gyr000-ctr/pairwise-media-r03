"""texsharp 核心 API：可分离锐化卷积（unsharp mask）。

设计要点
--------
* 使用归一化高斯核（和精确为 1，抽头均为二进制可精确表示的分数）做可分离
  模糊，再按 ``out = image + amount * (image - blur)`` 合成。
  等价卷积核 ``(1 + amount) * delta - amount * gaussian`` 的和恒为 1，
  因此常量图在任意 ``amount`` 下输出仍是同一常量，不产生振铃。
* 边界模式 ``clamp``（边缘复制）与 ``zero``（外部视作 0）产生不同结果。
* 可分离实现：水平、竖直各一次一维卷积，10 万像素远快于 300 ms。
* 参数校验区分“非法参数”并统一抛 ``ValueError``（含非有限像素/参数）。
* 无共享可变状态：同一实例可被多线程并发调用。
"""

from __future__ import annotations

import math
import numbers

__all__ = ["Sharpen", "KERNEL", "EDGE_MODES"]

#: 归一化 5 抽头高斯核（1/16, 4/16, 6/16, 4/16, 1/16），和精确为 1。
_TAPS = (1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0)

EDGE_MODES = ("clamp", "zero")


class _TypeValueError(TypeError, ValueError):
    """类型错误，同时属于 ``ValueError``：既按“类型不对”也按“非法值”归类。"""


#: 二维等价锐化核（和 = 1），供检查/展示，不参与计算。
KERNEL = tuple(tuple(wy * wx for wx in _TAPS) for wy in _TAPS)


class Sharpen:
    """锐化卷积器。无实例状态，可安全地在多线程间共享。"""

    def apply(self, image, amount=1.0, edge="clamp"):
        """锐化二维灰度图，返回新的 ``list[list[float]]``。

        参数
        ----
        image : 二维序列；像素须为有限实数，行宽一致且非空。
        amount : 锐化强度；``0.0`` 返回输入副本，须为有限实数。
        edge : ``"clamp"`` 或 ``"zero"``；其它值抛 ``ValueError``。
        """
        amount = _check_amount(amount)
        if not isinstance(edge, str) or edge not in EDGE_MODES:
            raise _TypeValueError(
                "edge must be one of %r, got %r" % (EDGE_MODES, edge)
            )
        grid = _validated_image(image)
        if amount == 0.0:
            return [row[:] for row in grid]

        blur = _blur(_blur(grid, edge, horizontal=True), edge, horizontal=False)
        keep = 1.0 + amount
        return [
            [keep * src - amount * b for src, b in zip(srow, brow)]
            for srow, brow in zip(grid, blur)
        ]


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #
def _is_finite_number(v) -> bool:
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        return False
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(f)


def _check_amount(amount) -> float:
    if not _is_finite_number(amount):
        raise _TypeValueError(
            "amount must be a finite real number, got %r" % (amount,)
        )
    return float(amount)


def _validated_image(image) -> list[list[float]]:
    if image is None or isinstance(image, (str, bytes, bytearray)):
        raise _TypeValueError("image must be a non-empty 2D sequence")
    try:
        rows = list(image)
    except TypeError:
        raise _TypeValueError("image must be an iterable of rows") from None
    if not rows:
        raise ValueError("image must not be empty")
    width = None
    grid: list[list[float]] = []
    for r, row in enumerate(rows):
        if row is None or isinstance(row, (str, bytes, bytearray)):
            raise _TypeValueError("row %d is not a sequence of numbers" % r)
        try:
            vals = list(row)
        except TypeError:
            raise _TypeValueError("row %d is not iterable" % r) from None
        if width is None:
            width = len(vals)
            if width == 0:
                raise ValueError("rows must not be empty")
        elif len(vals) != width:
            raise ValueError(
                "ragged image: row %d has %d values, expected %d"
                % (r, len(vals), width)
            )
        out_row = []
        for c, v in enumerate(vals):
            if not _is_finite_number(v):
                raise _TypeValueError(
                    "non-finite or non-numeric pixel at (%d, %d): %r"
                    % (r, c, v)
                )
            out_row.append(float(v))
        grid.append(out_row)
    return grid


# --------------------------------------------------------------------------- #
# 可分离模糊
# --------------------------------------------------------------------------- #
def _blur(grid, edge, horizontal):
    """对每个像素做一维 5 抽头归一化高斯模糊。"""
    radius = len(_TAPS) // 2
    height = len(grid)
    width = len(grid[0])
    taps = _TAPS
    zero = edge == "zero"
    out: list[list[float]] = []

    if horizontal:
        for row in grid:
            if zero:
                padded = [0.0] * radius + row + [0.0] * radius
            else:  # clamp：复制首/尾像素
                padded = [row[0]] * radius + row + [row[-1]] * radius
            acc = [0.0] * width
            for k, w in enumerate(taps):
                seg = padded[k:k + width]
                acc = [a + w * s for a, s in zip(acc, seg)]
            out.append(acc)
        return out

    # 竖直方向：按行平移，越界行在 clamp 下复制边缘行，在 zero 下跳过。
    out = [[0.0] * width for _ in range(height)]
    for k, w in enumerate(taps):
        offset = k - radius
        for y in range(height):
            ys = y + offset
            if 0 <= ys < height:
                src = grid[ys]
            elif zero:
                continue
            else:  # clamp
                src = grid[0 if ys < 0 else height - 1]
            acc = out[y]
            out[y] = [a + w * s for a, s in zip(acc, src)]
    return out
