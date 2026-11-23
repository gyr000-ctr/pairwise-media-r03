"""srgb2lab 核心 API（初始未实现）。"""


class SrgbLab:
    def to_lab(self, rgb):
        """转 Lab"""
        raise NotImplementedError

    def to_rgb(self, lab):
        """转 sRGB"""
        raise NotImplementedError
