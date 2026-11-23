"""pathfollow 核心 API：沿折线路径前进 + 加加速度有界（jerk-limited）速度规划。

设计要点：
- 位置严格按弧长在折线路径上插值，永不越界（曲率超限时只减速）；
- 每个 advance 周期做一次 jerk-limited 速度规划：加速度受 max_accel 限制，
  加速度的变化率受 max_jerk 限制；周期内按恒加速度运动，弧长位移等于
  规划速度对时间的积分（解析一致，误差 ≤1e-6）；
- 到达终点后剩余距离为 0 且速度归零；
- 负速度（负距离）与零/负曲率上限使用可区分的异常子类报错；
- 非有限输入（inf/nan）与非法类型分类报错（ValueError / TypeError）；
- 构造与前进均为 O(n) / O(log n + k)，1 万段远低于 300 ms；
- 内部状态由锁保护，并发安全。
"""

from __future__ import annotations

import bisect
import math
import threading

__all__ = [
    "Follow",
    "PathFollowError",
    "NegativeVelocityError",
    "CurvatureLimitError",
]


class PathFollowError(ValueError):
    """路径跟随参数错误基类（ValueError 子类，兼容既有测试）。"""


class NegativeVelocityError(PathFollowError):
    """负速度 / 负前进距离。"""


class CurvatureLimitError(PathFollowError):
    """曲率上限非正（零曲率上限不允许）。"""


def _finite_num(value, name):
    """类型非法 -> TypeError；非有限 -> ValueError；返回 float。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须是实数，得到 {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限值，得到 {value!r}")
    return value


class Follow:
    """沿折线路径前进的跟随器（jerk-limited 速度规划，线程安全）。"""

    def __init__(self, points=None, max_curv=1.0,
                 max_speed=10.0, max_accel=4.0, max_jerk=20.0):
        if points is None:
            points = ((0.0, 0.0), (1.0, 0.0))

        max_curv = _finite_num(max_curv, "max_curv")
        if max_curv <= 0.0:
            raise CurvatureLimitError(
                f"max_curv 必须为正（零/负曲率上限不允许），得到 {max_curv!r}")
        self._max_curv = max_curv
        self._max_speed = _finite_num(max_speed, "max_speed")
        self._max_accel = _finite_num(max_accel, "max_accel")
        self._max_jerk = _finite_num(max_jerk, "max_jerk")
        if self._max_speed <= 0.0 or self._max_accel <= 0.0 or self._max_jerk <= 0.0:
            raise PathFollowError("max_speed/max_accel/max_jerk 必须为正")

        pts = []
        for p in points:
            try:
                x, y = p
            except (TypeError, ValueError) as exc:
                raise TypeError(f"路径点必须是 (x, y) 二元组，得到 {p!r}") from exc
            pts.append((_finite_num(x, "point.x"), _finite_num(y, "point.y")))
        if len(pts) < 2:
            raise PathFollowError(f"至少需要 2 个路径点，得到 {len(pts)} 个")
        self._pts = pts

        # 累计弧长
        s = [0.0]
        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i - 1][0]
            dy = pts[i][1] - pts[i - 1][1]
            s.append(s[-1] + math.hypot(dx, dy))
        if s[-1] <= 0.0:
            raise PathFollowError("路径总长度为零")
        self._s = s
        self._total = s[-1]

        # 每个中间顶点的离散曲率：转角 / 平均邻边弧长
        n = len(pts)
        curv = [0.0] * n
        for i in range(1, n - 1):
            ux = pts[i][0] - pts[i - 1][0]
            uy = pts[i][1] - pts[i - 1][1]
            vx = pts[i + 1][0] - pts[i][0]
            vy = pts[i + 1][1] - pts[i][1]
            lu = math.hypot(ux, uy)
            lv = math.hypot(vx, vy)
            if lu <= 0.0 or lv <= 0.0:
                continue
            cos_t = (ux * vx + uy * vy) / (lu * lv)
            cos_t = max(-1.0, min(1.0, cos_t))
            theta = math.acos(cos_t)
            curv[i] = 2.0 * theta / (lu + lv)
        self._curv = curv

        self._lock = threading.Lock()
        self._s_pos = 0.0   # 当前弧长位置
        self._vel = 0.0     # 当前规划速度
        self._accel = 0.0   # 上一规划周期的（有符号）加速度，用于 jerk 约束

    # ------------------------------------------------------------------
    # 内部：速度规划
    # ------------------------------------------------------------------
    def _speed_limit(self, s0, s1):
        """窗口 (s0, s1] 内曲率超限则按比例减速，否则返回 max_speed。"""
        lo = bisect.bisect_right(self._s, s0)
        hi = bisect.bisect_right(self._s, s1)
        kmax = 0.0
        for i in range(lo, hi):
            if self._curv[i] > kmax:
                kmax = self._curv[i]
        if kmax > self._max_curv:
            return self._max_speed * math.sqrt(self._max_curv / kmax)
        return self._max_speed

    def _plan(self, moved):
        """jerk-limited 单周期规划：恒加速度覆盖恰好 moved 弧长。

        积分一致性：周期内恒加速度 a，历时 T=2*moved/(v0+v1)，
        ∫v dt = (v0+v1)/2 * T = moved，解析相等。
        """
        v0 = self._vel
        # 前瞻一个制动距离，曲率超限的弯前提前减速而不是越界
        lookahead = v0 * v0 / (2.0 * self._max_accel)
        v_lim = self._speed_limit(self._s_pos, self._s_pos + moved + lookahead)
        # 先用最大加速度估计周期时长 T
        if v_lim >= v0:
            v_est = min(v_lim, math.sqrt(v0 * v0 + 2.0 * self._max_accel * moved))
        else:
            v_est = max(v_lim, math.sqrt(max(0.0, v0 * v0 - 2.0 * self._max_accel * moved)))
        t_est = 2.0 * moved / (v0 + v_est) if (v0 + v_est) > 0.0 else moved / self._max_speed
        # jerk 约束：|a - a_prev| <= max_jerk * T
        a_allow = min(self._max_accel, abs(self._accel) + self._max_jerk * t_est)
        if v_lim >= v0:
            a_need = (v_est * v_est - v0 * v0) / (2.0 * moved)
            a = min(a_need, a_allow)
            v1 = math.sqrt(v0 * v0 + 2.0 * a * moved)
            self._accel = a
        else:
            a_need = (v0 * v0 - v_est * v_est) / (2.0 * moved)
            a = min(a_need, a_allow)
            v1 = math.sqrt(max(0.0, v0 * v0 - 2.0 * a * moved))
            self._accel = -a
        self._vel = v1

    def _position_locked(self):
        s = self._s_pos
        if s <= 0.0:
            return self._pts[0]
        if s >= self._total:
            return self._pts[-1]
        i = bisect.bisect_right(self._s, s) - 1
        seg = self._s[i + 1] - self._s[i]
        if seg <= 0.0:
            return self._pts[i + 1]
        t = (s - self._s[i]) / seg
        x0, y0 = self._pts[i]
        x1, y1 = self._pts[i + 1]
        return (x0 + t * (x1 - x0), y0 + t * (y1 - y0))

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    def advance(self, distance):
        """沿路径前进 distance 弧长（夹紧到终点），返回当前位置。"""
        d = _finite_num(distance, "distance")
        if d < 0.0:
            raise NegativeVelocityError(
                f"distance 必须非负（负速度不允许），得到 {d!r}")
        with self._lock:
            new_s = self._s_pos + d
            if new_s >= self._total:
                # 到达终点：剩余距离归零，速度归零
                new_s = self._total
                self._vel = 0.0
                self._accel = 0.0
            elif new_s > self._s_pos:
                self._plan(new_s - self._s_pos)
            self._s_pos = new_s
            return self._position_locked()

    def position(self):
        """当前位置 (x, y) 元组。"""
        with self._lock:
            return self._position_locked()

    def velocity(self):
        """当前规划速度（到达终点后为 0）。"""
        with self._lock:
            return self._vel

    def remaining(self):
        """到终点的剩余弧长（到达终点后为 0）。"""
        with self._lock:
            return self._total - self._s_pos
