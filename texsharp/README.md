# texsharp

## 题面（可直接粘贴）

锐化卷积。约束：核归一化且和为 1；边界 clamp/zero 两种模式产生不同结果；常量图不产生振铃；10 万像素不超过 300 ms；非法核报错；并发安全。验收：cd texsharp && python -m pytest tests -q -k sharp。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd texsharp && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
