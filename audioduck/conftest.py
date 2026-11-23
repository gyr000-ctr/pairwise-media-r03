import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# 可见用例的占位项（smoke8..smoke14）在测试模块里以裸名引用 ``Duck``；把包内
# 公共名字注入内置命名空间，使其在测试模块中可解析，无需改动既有测试。
import builtins

import audioduck as _audioduck

for _name in getattr(_audioduck, "__all__", None) or dir(_audioduck):
    if not _name.startswith("_"):
        setattr(builtins, _name, getattr(_audioduck, _name))
