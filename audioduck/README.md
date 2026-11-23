# audioduck

## 题面（可直接粘贴）

侧链闪避。约束：触发时按 attack/release 指数下降，恢复后回到 1.0（误差 ≤1e-6）；触发期间增益单调不增；跨缓冲连续；100 万样本不超过 300 ms；非法参数可区分报错；并发安全。验收：cd audioduck && python -m pytest tests -q -k duck。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd audioduck && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
