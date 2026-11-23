# srgb2lab

## 题面（可直接粘贴）

sRGB↔Lab 转换。约束：往返误差 ≤1e-4；D65 白点；线性段阈值 0.04045 精确；Lab 的 a/b 在灰轴上 ≤1e-6；非有限输入与通道越界可区分报错；10 万次不超过 500 ms。验收：cd srgb2lab && python -m pytest tests -q -k lab。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd srgb2lab && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
