# Occupancy Energy Anomaly Detector

一个用建筑占用人数校正每小时能耗、再标记异常残差的教学项目。它用前一段时间拟合按小时区分的线性基线，并在之后的时间段用 MAD 阈值标记偏离值；它不使用第三方机器学习库。

## 运行

需要 Python 3.9 或更高版本。

```bash
python occupancy_energy_anomaly_detector.py
python -m unittest discover -s tests -v
```

首次运行会生成 `data/building_hourly.csv`，后续可编辑该 CSV 或通过 `--input` 指定输入。必需列是 `timestamp,occupancy_count,energy_kwh`。示例额外包含 `scenario_truth`，它只用于生成数据的离线演示评分，不进入基线模型。默认以前 7 天训练、后续记录测试，可用 `--train-days` 改训练天数。输出在 `results/report.json`、`results/scored_residuals.csv` 和 `results/review_flags.csv`。

## 术语

- **占用归一化**：比较相似占用水平下的能耗，避免把“人多所以用电多”直接误判为异常。
- **残差**：实际能耗减去模型按占用与小时估算的能耗。
- **无监督异常标记**：模型不依赖异常标签来拟合，而是把偏离训练分布的读数标成候选项。
- **MAD**：中位数绝对偏差；本项目用训练期残差 MAD 与最低阈值共同决定标记范围。
- **时间留出测试**：按时间先后分训练期和测试期，避免未来记录参与过去基线的拟合。

## 数据和边界

示例数据用固定随机种子生成，包含一个虚构的能耗尖峰情景。评估分数只说明程序在这个构造样例上的表现，不能外推到真实建筑。模型只按“小时”和“人数”作很简化的线性校正，没有纳入天气、设备状态、节假日、空间差异或计量误差。标记需要设施人员结合运行记录复核，不能直接解释成浪费、故障或原因，也不应自动触发控制操作。
