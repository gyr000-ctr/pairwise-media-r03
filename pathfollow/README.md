# pathfollow

## 题面（可直接粘贴）

沿路径前进。约束：按弧长前进且速度恒定；曲率超过上限时减速而不是越界；到达终点后剩余距离为 0；负速度报错；1 万段不超过 300 ms；并发安全。验收：cd pathfollow && python -m pytest tests -q -k path。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd pathfollow && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：游戏开发
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
