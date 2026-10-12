# Occupancy Energy Anomaly Detector

Standard-library Python demo that fits an occupancy-aware hourly energy baseline and flags robust residual outliers on a later time split. Synthetic labels are used only for offline evaluation.

```bash
python occupancy_energy_anomaly_detector.py
python -m unittest discover -s tests -v
```

Full Chinese documentation and human-review limits: [README.zh-CN.md](README.zh-CN.md).
