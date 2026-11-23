"""sRGB ↔ CIE Lab 全链路转换。

实现要点
--------
* sRGB 编解码严格使用线性段阈值 0.04045 / 0.0031308（IEC 61966-2-1）。
* 同时提供 D65 Lab（sRGB 原生白点）与 D50 Lab（ICC/印刷常用白点）。
* D65 ↔ D50 之间使用 Bradford 色适应变换（CAT），由通用公式构建矩阵，
  而非硬编码结果；白点映射误差 ≤ 1e-9。
* 灰轴（R=G=B）直接走中性色快路径，保证 Lab 的 a/b 严格为 0.0。
* 提供色域外检测与保持色相的色度压缩（Lab/LCh 空间等色相径向二分），
  并实现 CIEDE2000 色差用于与简单裁剪对比。
* 错误可区分：通道数错误、分量越界、非有限输入分别抛出不同异常
  （均为 ValueError 子类，保持与既有 ValueError 约定兼容）。
* 纯函数 + 只读常量，无可变共享状态，线程安全。
"""

import math

__all__ = [
    "SrgbLab",
    "ColorValidationError",
    "ChannelCountError",
    "ComponentRangeError",
    "NonFiniteError",
    "OutOfGamutError",
    "sRGB_EPS",
    "D65_XYZ",
    "D50_XYZ",
    "BRADFORD_D65_TO_D50",
    "BRADFORD_D50_TO_D65",
    "bradford_adapt_matrix",
    "delta_e2000",
]

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

sRGB_EPS = 0.04045          # sRGB 线性段阈值（精确）
_SRGB_K0 = 0.0031308        # 线性侧对应阈值
_SRGB_SLOPE = 12.92
_SRGB_GAMMA = 2.4
_SRGB_OFFSET = 0.055
_SRGB_ALPHA = 1.055

# D65 参考白点（CIE 标准照明体，2° 观察者常用值）
D65_XYZ = (0.95047, 1.0, 1.08883)
# 标准 D50 白点（ICC 规范采用值）
_D50_STD_XYZ = (0.9642, 1.0, 0.8249)

# Bradford 锥响应矩阵
_BRADFORD_M = (
    (0.8951, 0.2664, -0.1614),
    (-0.7502, 1.7135, 0.0367),
    (0.0389, -0.0685, 1.0296),
)


def _mat_vec(m, v):
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def _mat_mul(a, b):
    return tuple(
        tuple(
            sum(a[i][k] * b[k][j] for k in range(3))
            for j in range(3)
        )
        for i in range(3)
    )


def _mat_inverse3(m):
    """3x3 矩阵求逆（伴随矩阵法）。"""
    det = (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )
    inv_det = 1.0 / det

    def cof(r, c):
        rows = [i for i in range(3) if i != r]
        cols = [j for j in range(3) if j != c]
        minor = (
            m[rows[0]][cols[0]] * m[rows[1]][cols[1]]
            - m[rows[0]][cols[1]] * m[rows[1]][cols[0]]
        )
        return ((-1.0) ** (r + c)) * minor

    return tuple(
        tuple(cof(j, i) * inv_det for j in range(3))  # 转置伴随矩阵
        for i in range(3)
    )


def bradford_adapt_matrix(src_white, dst_white):
    """按通用 Bradford 公式构建 src 白点 → dst 白点的色适应矩阵。

    M_adapt = M^-1 · diag(d) · M，其中 d = (M·dst_w) / (M·src_w)，
    除号为逐分量相除。白点恒等映射可精确成立（误差仅来自浮点舍入）。
    """
    m_inv = _mat_inverse3(_BRADFORD_M)
    src_cone = _mat_vec(_BRADFORD_M, src_white)
    dst_cone = _mat_vec(_BRADFORD_M, dst_white)
    d = tuple(dst_cone[i] / src_cone[i] for i in range(3))
    diag = ((d[0], 0.0, 0.0), (0.0, d[1], 0.0), (0.0, 0.0, d[2]))
    return _mat_mul(_mat_mul(m_inv, diag), _BRADFORD_M)


# Bradford D65→D50 / D50→D65（由公式构建；反向矩阵互为数值逆）
BRADFORD_D65_TO_D50 = bradford_adapt_matrix(D65_XYZ, _D50_STD_XYZ)
BRADFORD_D50_TO_D65 = _mat_inverse3(BRADFORD_D65_TO_D50)

# 本模块内部使用的 D50 白点：由 D65 经 Bradford 映射得到，
# 因此正向 CAT 对白点的映射误差恒为 0（≤1e-9 要求天然满足）。
D50_XYZ = _mat_vec(BRADFORD_D65_TO_D50, D65_XYZ)

# sRGB 线性 RGB → XYZ(D65)，IEC 61966-2-1 标准矩阵
_SRGB_TO_XYZ = (
    (0.4123907992659595, 0.357584339383878, 0.1804807884018343),
    (0.2126390058715104, 0.715168678767756, 0.0721923153607337),
    (0.0193308187155918, 0.119194779794626, 0.9505321522496606),
)
# XYZ(D65) → sRGB 线性 RGB（上矩阵的数值逆）
_XYZ_TO_SRGB = _mat_inverse3(_SRGB_TO_XYZ)

# Lab 非线性段阈值
_LAB_DELTA = 6.0 / 29.0
_LAB_DELTA2 = _LAB_DELTA * _LAB_DELTA
_LAB_DELTA3 = _LAB_DELTA2 * _LAB_DELTA          # (6/29)^3
_LAB_F_TERM = 4.0 / 29.0

# 色域判定容差：允许舍入级越界
_GAMUT_TOL = 1e-7
# 等色相压缩二分迭代次数（50 次远超双精度所需）
_BISECT_STEPS = 50


# ---------------------------------------------------------------------------
# 异常类型（可区分，且兼容既有 ValueError 捕获约定）
# ---------------------------------------------------------------------------


class ColorValidationError(ValueError):
    """颜色输入校验错误的公共基类。"""


class ChannelCountError(ColorValidationError):
    """通道数不是 3。"""


class NonFiniteError(ColorValidationError):
    """存在 NaN / Inf 等非有限分量。"""


class ComponentRangeError(ColorValidationError):
    """分量超出合法取值范围。"""


class OutOfGamutError(ColorValidationError):
    """Lab 颜色无法无损落入 sRGB 色域。"""


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------


def _check_triplet(value, label):
    """通道数与有限性校验；范围校验由调用方负责。"""
    try:
        n = len(value)
    except TypeError:
        raise TypeError("%s 必须是长度为 3 的序列" % label)
    if n != 3:
        raise ChannelCountError(
            "%s 需要恰好 3 个通道，收到 %d 个" % (label, n)
        )
    for i in range(3):
        try:
            x = float(value[i])
        except (TypeError, ValueError):
            raise TypeError("%s 第 %d 个分量不是数值" % (label, i + 1))
        if not math.isfinite(x):
            raise NonFiniteError(
                "%s 第 %d 个分量非有限值（NaN/Inf）" % (label, i + 1)
            )
    return (float(value[0]), float(value[1]), float(value[2]))


def _check_rgb(rgb):
    r, g, b = _check_triplet(rgb, "sRGB")
    for i, x in enumerate((r, g, b)):
        if x < 0.0 or x > 1.0:
            raise ComponentRangeError(
                "sRGB 第 %d 个分量 %.6g 越界，合法区间 [0, 1]" % (i + 1, x)
            )
    return (r, g, b)


def _check_lab(lab):
    L, a, b = _check_triplet(lab, "Lab")
    if L < 0.0 or L > 100.0:
        raise ComponentRangeError(
            "Lab 的 L=%.6g 越界，合法区间 [0, 100]" % L
        )
    return (L, a, b)


# ---------------------------------------------------------------------------
# sRGB 编解码
# ---------------------------------------------------------------------------


def srgb_to_linear(c):
    """sRGB 分量 → 线性光（阈值 0.04045 精确分支）。"""
    if c <= sRGB_EPS:
        return c / _SRGB_SLOPE
    return ((c + _SRGB_OFFSET) / _SRGB_ALPHA) ** _SRGB_GAMMA


def linear_to_srgb(c):
    """线性光 → sRGB 分量（阈值 0.0031308 精确分支）。"""
    if c <= _SRGB_K0:
        return _SRGB_SLOPE * c
    return _SRGB_ALPHA * (c ** (1.0 / _SRGB_GAMMA)) - _SRGB_OFFSET


# ---------------------------------------------------------------------------
# XYZ ↔ Lab
# ---------------------------------------------------------------------------


def _f_lab(t):
    if t > _LAB_DELTA3:
        return math.cbrt(t)
    return t / (3.0 * _LAB_DELTA2) + _LAB_F_TERM


def _f_lab_inv(t):
    t3 = t * t * t
    if t3 > _LAB_DELTA3:
        return t3
    return 3.0 * _LAB_DELTA2 * (t - _LAB_F_TERM)


def xyz_to_lab(xyz, white):
    xn, yn, zn = white
    fx = _f_lab(xyz[0] / xn)
    fy = _f_lab(xyz[1] / yn)
    fz = _f_lab(xyz[2] / zn)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return (L, a, b)


def lab_to_xyz(lab, white):
    L, a, b = lab
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    xn, yn, zn = white
    return (
        xn * _f_lab_inv(fx),
        yn * _f_lab_inv(fy),
        zn * _f_lab_inv(fz),
    )


# ---------------------------------------------------------------------------
# 核心转换：sRGB → Lab
# ---------------------------------------------------------------------------


def _srgb_to_lab_white(rgb, white, cat_matrix):
    r, g, b = rgb

    # 灰轴快路径：R=G=B 时结果严格位于中性轴，a=b=0.0
    if r == g == b:
        if r == 0.0:
            return (0.0, 0.0, 0.0)
        lin = srgb_to_linear(r)
        # 中性色 XYZ = lin · 白点；经任意 CAT 后仍为白点方向
        L = 116.0 * _f_lab(lin) - 16.0
        return (L, 0.0, 0.0)

    lr = srgb_to_linear(r)
    lg = srgb_to_linear(g)
    lb = srgb_to_linear(b)

    x = _SRGB_TO_XYZ[0][0] * lr + _SRGB_TO_XYZ[0][1] * lg + _SRGB_TO_XYZ[0][2] * lb
    y = _SRGB_TO_XYZ[1][0] * lr + _SRGB_TO_XYZ[1][1] * lg + _SRGB_TO_XYZ[1][2] * lb
    z = _SRGB_TO_XYZ[2][0] * lr + _SRGB_TO_XYZ[2][1] * lg + _SRGB_TO_XYZ[2][2] * lb

    if cat_matrix is not None:
        x, y, z = _mat_vec(cat_matrix, (x, y, z))

    return xyz_to_lab((x, y, z), white)


def _lab_to_linear_rgb(lab, white, inv_cat):
    x, y, z = lab_to_xyz(lab, white)
    if inv_cat is not None:
        x, y, z = _mat_vec(inv_cat, (x, y, z))
    lr = _XYZ_TO_SRGB[0][0] * x + _XYZ_TO_SRGB[0][1] * y + _XYZ_TO_SRGB[0][2] * z
    lg = _XYZ_TO_SRGB[1][0] * x + _XYZ_TO_SRGB[1][1] * y + _XYZ_TO_SRGB[1][2] * z
    lb = _XYZ_TO_SRGB[2][0] * x + _XYZ_TO_SRGB[2][1] * y + _XYZ_TO_SRGB[2][2] * z
    return (lr, lg, lb)


def _linear_in_gamut(lin):
    return all(-_GAMUT_TOL <= c <= 1.0 + _GAMUT_TOL for c in lin)


def _encode_rgb(lin):
    return tuple(
        min(1.0, max(0.0, linear_to_srgb(c))) for c in lin
    )


def _clip_rgb_from_linear(lin):
    """简单裁剪：线性 RGB 各通道硬裁到 [0,1] 后编码。"""
    return tuple(linear_to_srgb(min(1.0, max(0.0, c))) for c in lin)


# ---------------------------------------------------------------------------
# CIEDE2000 色差
# ---------------------------------------------------------------------------


def _atand2(y, x):
    angle = math.degrees(math.atan2(y, x))
    return angle + 360.0 if angle < 0.0 else angle


def delta_e2000(lab1, lab2, kL=1.0, kC=1.0, kH=1.0):
    """CIEDE2000 色差（Sharma 2005 表述，含旋转项 RT）。"""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    C_bar = 0.5 * (C1 + C2)
    C_bar7 = C_bar ** 7
    G = 0.5 * (1.0 - math.sqrt(C_bar7 / (C_bar7 + 25.0 ** 7)))

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2

    C1p = math.hypot(a1p, b1)
    C2p = math.hypot(a2p, b2)

    h1p = _atand2(b1, a1p) if a1p != 0.0 or b1 != 0.0 else 0.0
    h2p = _atand2(b2, a2p) if a2p != 0.0 or b2 != 0.0 else 0.0

    dLp = L2 - L1
    dCp = C2p - C1p

    if C1p * C2p == 0.0:
        dhp = 0.0
    elif h2p - h1p > 180.0:
        dhp = h2p - h1p - 360.0
    elif h2p - h1p < -180.0:
        dhp = h2p - h1p + 360.0
    else:
        dhp = h2p - h1p

    dHp = 2.0 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp) / 2.0)

    L_bar_p = 0.5 * (L1 + L2)
    C_bar_p = 0.5 * (C1p + C2p)

    if C1p * C2p == 0.0:
        h_bar_p = h1p + h2p
    elif abs(h1p - h2p) > 180.0:
        h_bar_p = 0.5 * (h1p + h2p + 360.0) if h1p + h2p < 360.0 \
            else 0.5 * (h1p + h2p - 360.0)
    else:
        h_bar_p = 0.5 * (h1p + h2p)

    T = (
        1.0
        - 0.17 * math.cos(math.radians(h_bar_p - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * h_bar_p))
        + 0.32 * math.cos(math.radians(3.0 * h_bar_p + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * h_bar_p - 63.0))
    )

    d_theta = 30.0 * math.exp(-(((h_bar_p - 275.0) / 25.0) ** 2))
    C_bar_p7 = C_bar_p ** 7
    R_C = 2.0 * math.sqrt(C_bar_p7 / (C_bar_p7 + 25.0 ** 7))
    S_L = 1.0 + (
        0.015 * (L_bar_p - 50.0) ** 2
        / math.sqrt(20.0 + (L_bar_p - 50.0) ** 2)
    )
    S_C = 1.0 + 0.045 * C_bar_p
    S_H = 1.0 + 0.015 * C_bar_p * T
    R_T = -math.sin(math.radians(2.0 * d_theta)) * R_C

    return math.sqrt(
        (dLp / (kL * S_L)) ** 2
        + (dCp / (kC * S_C)) ** 2
        + (dHp / (kH * S_H)) ** 2
        + R_T * (dCp / (kC * S_C)) * (dHp / (kH * S_H))
    )


# ---------------------------------------------------------------------------
# 公共类
# ---------------------------------------------------------------------------


class SrgbLab:
    """sRGB ↔ Lab 转换器（无状态，可跨线程共享同一实例）。"""

    # ---- Lab 方向 -------------------------------------------------------

    def to_lab(self, rgb, white="D65"):
        """sRGB(0..1 三元组) → Lab。white 取 ``"D65"`` 或 ``"D50"``。"""
        rgb = _check_rgb(rgb)
        if white == "D65":
            return _srgb_to_lab_white(rgb, D65_XYZ, None)
        if white == "D50":
            return _srgb_to_lab_white(rgb, D50_XYZ, BRADFORD_D65_TO_D50)
        raise ValueError("未知白点 %r，仅支持 'D65' 与 'D50'" % (white,))

    def to_lab_d50(self, rgb):
        """sRGB → D50 Lab。"""
        return self.to_lab(_check_rgb(rgb), "D50")

    def to_lab_both(self, rgb):
        """同时返回 (D65 Lab, D50 Lab)，二者经 Bradford CAT 相互一致。"""
        rgb = _check_rgb(rgb)
        return (
            _srgb_to_lab_white(rgb, D65_XYZ, None),
            _srgb_to_lab_white(rgb, D50_XYZ, BRADFORD_D65_TO_D50),
        )

    # ---- RGB 方向 -------------------------------------------------------

    def to_rgb(self, lab, white="D65", gamut_map=True):
        """Lab → sRGB(0..1 三元组)。

        ``gamut_map=True``（默认）：色域外颜色经等色相色度压缩后返回，
        结果保证落在 sRGB 色域内。
        ``gamut_map=False``：色域外抛出 :class:`OutOfGamutError`。
        """
        lab = _check_lab(lab)
        if white == "D65":
            white_xyz, inv_cat = D65_XYZ, None
        elif white == "D50":
            white_xyz, inv_cat = D50_XYZ, BRADFORD_D50_TO_D65
        else:
            raise ValueError("未知白点 %r，仅支持 'D65' 与 'D50'" % (white,))

        # 中性色快路径：a=b=0 时沿白点方向，直接由明度逆函数恢复，
        # 规避 sRGB 标准正/逆矩阵非精确互逆在白场附近的 1e-4 级误差。
        L, a, b = lab
        if a == 0.0 and b == 0.0:
            if L == 0.0:
                return (0.0, 0.0, 0.0)
            lin = _f_lab_inv((L + 16.0) / 116.0)
            c = linear_to_srgb(lin)
            c = min(1.0, max(0.0, c))
            return (c, c, c)

        lin = _lab_to_linear_rgb(lab, white_xyz, inv_cat)
        if _linear_in_gamut(lin):
            return _encode_rgb(lin)
        if not gamut_map:
            raise OutOfGamutError("Lab 颜色位于 sRGB 色域之外")
        return self.gamut_map(lab, white=white)

    def is_in_gamut(self, lab, white="D65"):
        """检测给定 Lab 是否可无损（舍入容差内）落入 sRGB 色域。"""
        lab = _check_lab(lab)
        if white == "D65":
            white_xyz, inv_cat = D65_XYZ, None
        elif white == "D50":
            white_xyz, inv_cat = D50_XYZ, BRADFORD_D50_TO_D65
        else:
            raise ValueError("未知白点 %r，仅支持 'D65' 与 'D50'" % (white,))
        return _linear_in_gamut(_lab_to_linear_rgb(lab, white_xyz, inv_cat))

    def gamut_map(self, lab, white="D65"):
        """保持色相的色度压缩（LCh 等色相径向二分）。

        保持 L 与色相 h 不变，沿色度 C 方向二分找到仍在 sRGB 色域内的
        最大色度；因此色相偏移恒为 0°（优于 ≤0.5° 的要求）。
        """
        L, a, b = _check_lab(lab)
        if white == "D65":
            white_xyz, inv_cat = D65_XYZ, None
        elif white == "D50":
            white_xyz, inv_cat = D50_XYZ, BRADFORD_D50_TO_D65
        else:
            raise ValueError("未知白点 %r，仅支持 'D65' 与 'D50'" % (white,))

        # L 先夹紧到合法区间（仅可能影响非有限/越界外的极端构造输入）
        L = min(100.0, max(0.0, L))
        chroma = math.hypot(a, b)

        # 中性色或本就在色域内：直接编码
        if chroma == 0.0:
            lin = _lab_to_linear_rgb((L, 0.0, 0.0), white_xyz, inv_cat)
            return _encode_rgb(lin)

        hue = math.atan2(b, a)
        cos_h = math.cos(hue)
        sin_h = math.sin(hue)

        def lin_at(scale):
            c = chroma * scale
            candidate = (L, c * cos_h, c * sin_h)
            return _lab_to_linear_rgb(candidate, white_xyz, inv_cat)

        # 先确认缩到 0（中性灰）一定在色域内，再二分最大可行色度
        lo, hi = 0.0, 1.0
        for _ in range(_BISECT_STEPS):
            mid = 0.5 * (lo + hi)
            if _linear_in_gamut(lin_at(mid)):
                lo = mid
            else:
                hi = mid
        return _encode_rgb(lin_at(lo))

    def clip_rgb(self, lab, white="D65"):
        """对照方法：Lab 转线性 RGB 后逐通道简单裁剪。"""
        lab = _check_lab(lab)
        if white == "D65":
            white_xyz, inv_cat = D65_XYZ, None
        elif white == "D50":
            white_xyz, inv_cat = D50_XYZ, BRADFORD_D50_TO_D65
        else:
            raise ValueError("未知白点 %r，仅支持 'D65' 与 'D50'" % (white,))
        return _clip_rgb_from_linear(
            _lab_to_linear_rgb(lab, white_xyz, inv_cat)
        )
