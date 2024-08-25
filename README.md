<p align="center">
<a href="https://github.com/google/yapf/actions/workflows/pre-commit.yml"><img alt="Actions Status" src="https://github.com/google/yapf/actions/workflows/pre-commit.yml/badge.svg"></a>
</p>


<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

![framework](./resources/framework.svg)

## Introduction

STD EvalKit (Small Target Detection Evaluation Kit) is a library of evaluation metrics toolbox for infrared small target segmentation tasks.

We statistics the evaluation metrics in the field of infrared small target segmentation in recent years([statistical results](https://github.com/IRSTD/StatOnEvalMetrics)).


<details open>
<summary>Major features</summary>

- **High Efficiency**

    Multi-threading.

- **Device Friendly**

    All metrics support automatic batch accumulation.

- **Unified API**

    All metrics provide the same API, `Metric.update(labels, preds)` complete the accumulation of batches， `Metric.get()` get metrics。

- **Unified Computational**

    We use the same calculation logic and algorithms for the same type of metrics, ensuring consistency between results.

- **Supports multiple data formats**

    Supports multiple input data formats, hwc/chw/bchw/bhwc/image path, more details in <div> <a href="./notebook/tutorials.ipynb">./notebook/tutorial.ipynb</a></div>


</details>

## Overview of Metrics

Based on the data required for the calculation of the evaluation metrics, we have classified the metrics into three broad categories, Pixel-Level, Center-Level, and Center.

<div align="center">
  <b>Architectures</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Pixel Level</b>
      </td>
      <td colspan="2">
        <b>Target Level</b>
      </td>
    </tr>
    <tr valign="top" valign="bottom">
      <td rowspan="2">
        <ul>
            <li><a href="stdeval/metrics/pixel_level/pixel_auc_roc_ap_pr.py">AUC ROC AP PR</a></li>
            <li><a href="stdeval/metrics/pixel_level/pixel_pre_rec_f1_iou.py">Precision Recall F1 IoU (DOI:10.1109/TAES.2023.3238703)</a></li>
            <li><a href="stdeval/metrics/pixel_level/pixel_normalized_iou.py">NormalizedIoU (DOI:10.1109/WACV48630.2021.00099)</a></li>
      </ul>
      </td>
        <td align="center"><b>Center-Level</b></td>
        <td align="center"><b>Box Level</b></td>
    <tr valign="top">
      <td>
        <ul>
          <li><a href="stdeval/metrics/target_level/center_level/center_pre_rec_f1.py">Precision Recall F1 (DOI:10.1109/TAES.2022.3159308)</a></li>
                    <li><a href="stdeval/metrics/target_level/center_level/center_ap.py">Average Precision (Ours)</a></li>
            <li><a href="stdeval/metrics/target_level/center_level/center_pd_fa.py">Pd_Fa (DOI:10.1109/TIP.2022.3199107)</a></li>
            <li><a href="stdeval/metrics/target_level/center_level/center_roc_pd_fa.py">ROC Pd_Fa</a></li>
            <li><a href="stdeval/metrics/target_level/center_level/center_normalized_iou.py">Center Normalized IoU (Ours)</a></li>
        </ul>
      </td>
      <td>
        <ul>
            <li><a href="stdeval/metrics/target_level/box_level/box_mean_ap_ar.py">Mean Average Precision, Recall (COCO)</a></li>
        </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

## Installation
```bash
git clone git@github.com:IRSTD/STD-EvalKit.git
```
```bash
cd STD-EvalKit
```
For developers(recommended, easy for debugging)
```bash
pip install -e .
```
Only use
```bash
pip install stdeval
```


## Tutorial
```python
from stdeval.metrics import PixelPrecisionRecallF1IoU
Metric = PixelPrecisionRecallF1IoU(
    conf_thr=0.5,
    )
Metric.update(labels=labels, preds=preds.sigmoid())
precision, recall, f1_score, iou = Metric.get()
```
For more details, please refer to <div><a href="./notebook/tutorials.ipynb">./notebook/tutorial.ipynb</a></div>
