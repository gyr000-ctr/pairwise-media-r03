"""pathfollow 核心 API（初始未实现）。"""


class Follow:
    def __init__(self, points, max_curv):
        """构造"""
        raise NotImplementedError

    def advance(self, distance):
        """前进"""
        raise NotImplementedError

    def position(self):
        """当前位置"""
        raise NotImplementedError
