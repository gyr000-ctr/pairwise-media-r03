import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


def pytest_collection_modifyitems(items):
    """兼容既有占位测试：其 test_spec 回退分支引用了裸名 SrgbLab，
    而测试模块只做了 ``import srgb2lab``。在不改测试文件的前提下，
    于收集期把该符号注入测试模块的全局命名空间。"""
    for item in items:
        func = getattr(item, "function", None)
        if func is None:
            continue
        globs = getattr(func, "__globals__", None)
        if globs is not None and "SrgbLab" not in globs:
            import srgb2lab
            if hasattr(srgb2lab, "SrgbLab"):
                globs["SrgbLab"] = srgb2lab.SrgbLab
