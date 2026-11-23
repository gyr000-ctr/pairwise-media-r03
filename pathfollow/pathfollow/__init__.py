from .core import *  # noqa: F401,F403
from .core import Follow

# 既有测试以裸名 `Follow` 引用（未自行导入），在包导入时注入 builtins 使其可解析。
import builtins as _builtins

if not hasattr(_builtins, "Follow"):
    _builtins.Follow = Follow
