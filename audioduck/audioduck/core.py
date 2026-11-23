"""audioduck 核心 API（初始未实现）。"""


class Duck:
    def __init__(self, depth_db, attack, release):
        """构造"""
        raise NotImplementedError

    def process(self, samples):
        """处理"""
        raise NotImplementedError

    def gain(self):
        """当前增益"""
        raise NotImplementedError
