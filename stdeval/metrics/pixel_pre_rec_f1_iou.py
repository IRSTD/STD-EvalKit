import threading
from typing import Any

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix

from .base import BaseMetric, time_cost_deco
from .utils import _TYPES, _safe_divide, convert2format


class PixelPrecisionRecallF1IoU(BaseMetric):

    def __init__(self, conf_thr: float = 0.5, **kwargs: Any):
        """Pixel Level.
            Calculate Precision, Recall, F1, IoU for a given confidence threshold.

        Paper:
            @ARTICLE{10024907,
            author={Zhang, Tianfang and Li, Lei and Cao, Siying and Pu, Tian and Peng, Zhenming},
            journal={IEEE Transactions on Aerospace and Electronic Systems},
            title={Attention-Guided Pyramid Context Networks for Detecting \
                Infrared Small Target Under Complex Background},
            year={2023},
            volume={59},
            number={4},
            pages={4250-4261},
            keywords={Semantics;Object detection;Feature extraction;Correlation;Fuses;\
                Task analysis;Neural networks;Context module;feature fusion;\
                    infrared small targets;pyramid context network},
            doi={10.1109/TAES.2023.3238703}}

        Args:
            conf_thr (float, optional): Confidence threshold. Defaults to 0.5.
        """
        super().__init__(**kwargs)
        self.conf_thr = conf_thr
        self.lock = threading.Lock()
        self.reset()

    @time_cost_deco
    def update(self, labels: _TYPES, preds: _TYPES) -> None:

        def evaluate_worker(self, label: np.array, pred: np.array):
            tn, fp, fn, tp = self._confusion_mat(label, pred, self.conf_thr)
            with self.lock:
                self.tp[0] += tp
                self.fp[0] += fp
                self.fn[0] += fn
                self.tn[0] += tn

        labels, preds = convert2format(labels, preds)
        if isinstance(labels, np.ndarray):
            evaluate_worker(self, labels, preds)

        elif isinstance(labels, (list, tuple)):
            threads = [
                threading.Thread(
                    target=evaluate_worker,
                    args=(self, labels[i], preds[i]),
                ) for i in range(len(labels))
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            raise NotImplementedError

    @time_cost_deco
    def get(self):
        self.recall = _safe_divide(self.tp, self.tp + self.fn)
        self.precision = _safe_divide(self.tp, self.tp + self.fp)
        self.iou = _safe_divide(self.tp, self.tp + self.fp + self.fn)
        self.f1_score = _safe_divide(2 * self.precision * self.recall,
                                     self.precision + self.recall)
        if self.print_table:
            head = [
                f'Precision-{self.conf_thr}', f'Recall-{self.conf_thr}',
                f'F1-{self.conf_thr}', f'IOU-{self.conf_thr}'
            ]
            table = PrettyTable(head)
            table.add_row([
                '{:.4f}'.format(self.precision[0]),
                '{:.4f}'.format(self.recall[0]),
                '{:.4f}'.format(self.f1_score[0]), '{:.5f}'.format(self.iou[0])
            ])
            print(table)

        return self.precision, self.recall, self.f1_score, self.iou

    def reset(self):
        self.tp = np.zeros((1))
        self.fp = np.zeros((1))
        self.fn = np.zeros((1))
        self.tn = np.zeros((1))
        self.f1_score = np.zeros((1))
        self.iou = np.zeros((1))
        self.recall = np.zeros((1))
        self.precision = np.zeros((1))

    def _confusion_mat(self, label: np.array, pred: np.array,
                       score_thresh: float):
        predict = (pred > score_thresh).astype('float32').flatten()
        target = label.astype('int64').flatten()  # T
        tn, fp, fn, tp = confusion_matrix(target, predict).ravel()
        return tn, fp, fn, tp

    @property
    def table(self):
        all_metric = np.stack(
            [self.precision, self.recall, self.f1_score, self.iou]).T
        df = pd.DataFrame(all_metric)
        df.columns = [
            f'Precision-{self.conf_thr}', f'Recall-{self.conf_thr}',
            f'F1-{self.conf_thr}', f'IOU-{self.conf_thr}'
        ]
        return df
