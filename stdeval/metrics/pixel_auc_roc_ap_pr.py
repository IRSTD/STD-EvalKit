import threading
from typing import Any, List, Union

import numpy as np
import pandas as pd
import torch
from prettytable import PrettyTable
from sklearn.metrics import auc
from torchmetrics.classification import (BinaryAveragePrecision,
                                         BinaryPrecisionRecallCurve, BinaryROC)

from .base import BaseMetric, time_cost_deco
from .utils import _TYPES, convert2format


# codespell:ignore fpr
class PixelROCPrecisionRecall(BaseMetric):

    def __init__(self,
                 conf_thrs: Union[int, List[float], np.ndarray] = 10,
                 **kwargs: Any):
        """Pixel Level Curve.
        Calculate the curve of Precision, Recall, AP, AUC ROC for a given set of confidence thresholds.
        length of tpr, fpr are conf_thrs+1.

        .get() will return auc_roc, auc_pr, fpr, tpr, precision,
            recall in array.
        Args:
            conf_thrs (Union[int, List[float], np.ndarray]): Confidence thresholds.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced \
                from 0 to 1 as conf_thrs for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as conf_thrs \
                for the calculation
            - If set to an 1d `array` of floats, will use the indicated thresholds in the array as
                conf_thrs for the calculation.
        """
        super().__init__(**kwargs)
        self.conf_thrs = conf_thrs
        self.lock = threading.Lock()
        self.roc_curve_fn = BinaryROC(thresholds=self.conf_thrs)
        self.pr_curve_fn = BinaryPrecisionRecallCurve(
            thresholds=self.conf_thrs)
        # Average precision is not equal to auc_pr. This is due to the way the calculations are made
        self.ap_fn = BinaryAveragePrecision(thresholds=self.conf_thrs)
        self.reset()

    @time_cost_deco
    def update(self, labels: _TYPES, preds: _TYPES) -> None:

        def evaluate_worker(self, label: np.array, pred: np.array):
            ten_pred = torch.from_numpy(pred).to(torch.float32)
            ten_gt = torch.from_numpy(label).to(torch.int64)
            self.roc_curve_fn.update(ten_pred, ten_gt)
            self.pr_curve_fn.update(ten_pred, ten_gt)
            self.ap_fn.update(ten_pred, ten_gt)

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

        self.fpr, self.tpr, _ = self.roc_curve_fn.compute()
        self.precision, self.recall, _ = self.pr_curve_fn.compute()
        self.auc_roc = auc(self.fpr, self.tpr)
        self.auc_pr = auc(self.recall, self.precision)
        self.ap = self.ap_fn.compute().numpy()

        if self.print_table:
            head = [
                'AUC_ROC', 'AUC_PR(AUC function)',
                'AP(BinaryAveragePrecision function)'
            ]
            table = PrettyTable(head)
            table.add_row([self.auc_roc, self.auc_pr, self.ap])
            print(table)

        return self.auc_roc, self.auc_pr, self.fpr.numpy(), self.tpr.numpy(
        ), self.precision.numpy(), self.recall.numpy(), self.ap

    def reset(self):
        self.roc_curve_fn.reset()
        self.pr_curve_fn.reset()
        self.auc_roc = 0
        self.auc_pr = 0
        self.ap = 0

    @property
    def table(self):
        all_metric = np.stack([self.auc_roc, self.auc_pr,
                               self.ap])[:, np.newaxis].T
        df = pd.DataFrame(all_metric)
        df.columns = ['AUC_ROC', 'AUC_PR', 'AP']
        return df
