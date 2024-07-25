import threading
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
from prettytable import PrettyTable

from .base import BaseMetric, time_cost_deco
from .utils import (_TYPES, _adjust_dis_thr_arg, _safe_divide,
                    calculate_target_infos, convert2format,
                    get_label_coord_and_gray, get_pred_coord_and_gray,
                    second_match_method)


class TargetPrecisionRecallF1(BaseMetric):

    def __init__(self,
                 conf_thr: float = 0.5,
                 dis_thrs: Union[List[int], int] = [1, 10],
                 match_alg: str = 'forloop',
                 second_match: str = 'none',
                 **kwargs: Any):
        """Target-Level.
        TP: True Positive, GT is Positive and Pred is Positive, If Euclidean Distance < threshold, matched.
        FN: False Negative, GT is Positive and Pred is Negative.
        FP: False Positive, GT is Negative and Pred is Positive. If Euclidean Distance > threshold, not matched.
        Recall: TP/(TP+FN).
        Precision: TP/(TP+FP).
        F1: 2*Precision*Recall/(Precision+Recall).
        .get will return Precision, Recall, F1 in array.

        Args:
            conf_thr (float, Optional): Confidence threshold. Defaults to 0.5.
            dis_thrs (Union[List[int], int], optional):Threshold of euclidean distance for match gt and pred, \
                1 means start point, 10 means endpoints, includes endpoints. Defaults to [1, 10].
                - If set to an `int` , will use this value to distance threshold.
                - If set to an `list` of float or int, will use the indicated thresholds \
                    in the list as dis_thrs for the calculation
                - If set to an 1d `array` of floats, will use the indicated thresholds in the array as
                    dis_thrs for the calculation.
            match_alg (str, optional):'forloop' to match pred and gt,
                'forloop'is the original implementation of PD_FA,
                based on the first-match principle. Defaults to 'forloop'
            second_match (str, optional): Second match algorithm for match pred and gt after distance matching. \
                Support 'none', 'mask' and 'bbox'. 'none' means no secondary matching. Defaults to 'none'.
        """

        super().__init__(**kwargs)
        self.dis_thrs = _adjust_dis_thr_arg(dis_thrs)
        self.conf_thr = np.array([conf_thr])
        self.match_alg = match_alg
        self.second_match = second_match
        self.lock = threading.Lock()
        self.reset()

    @time_cost_deco
    def update(self, labels: _TYPES, preds: _TYPES) -> None:
        """Support CHW, BCHW, HWC,BHWC, Image Path, or in their list form (except BHWC/BCHW),
            like [CHW, CHW, ...], [HWC, HWC, ...], [Image Path, Image Path, ...].

            Although support Image Path, but not recommend.
            Note : All preds are probabilities image from 0 to 1 in default.
            If images, Preds must be probability image from 0 to 1 in default.
            If path, Preds must be probabilities image from 0-1 in default, if 0-255,
            we are force /255 to 0-1 to probability image.
        Args:
            labels (_TYPES): Ground Truth images or image paths in list or single.
            preds (_TYPES): Preds images or image paths in list or single.
        """

        def evaluate_worker(self, label, pred):
            # to unit8 for ``convert2gray()``
            coord_label, gray_label = get_label_coord_and_gray(label)
            coord_pred, gray_pred = get_pred_coord_and_gray(
                pred.copy(), self.conf_thr)
            distances, mask_iou, bbox_iou = calculate_target_infos(
                coord_label, coord_pred, gray_pred.shape[0],
                gray_pred.shape[1])
            if self.debug:
                print(f'bbox_iou={bbox_iou}')
                print(f'mask_iou={mask_iou}')
                print(f'eul_distance={distances}')
                print('____' * 20)

            if self.second_match != 'none':
                distances = second_match_method(distances, mask_iou, bbox_iou,
                                                self.second_match)
                if self.debug:
                    print(f'After second match eul distances={distances}')
                    print('____' * 20)

            for idx, threshold in enumerate(self.dis_thrs):
                TP, FN, FP = self._calculate_tp_fn_fp(distances.copy(),
                                                      threshold)
                with self.lock:
                    self.TP[idx] += TP
                    self.FP[idx] += FP
                    self.FN[idx] += FN
            return

        labels, preds = convert2format(labels, preds)

        if self.debug:
            for i in range(len(labels)):
                evaluate_worker(self, labels[i], preds[i])
        else:
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

    @time_cost_deco
    def get(self):
        """Compute metric

        Returns:
            _type_: Precision, Recall, micro-F1, shape == [1, num_threshold].
        """
        self._calculate_precision_recall_f1()
        head = ['Dis-Thr']
        head.extend(self.dis_thrs.tolist())
        table = PrettyTable()
        table.add_column('Dis-Thr', self.dis_thrs)
        table.add_column('TP', ['{:.0f}'.format(num) for num in self.TP])
        table.add_column('FP', ['{:.0f}'.format(num) for num in self.FP])
        table.add_column('FN', ['{:.0f}'.format(num) for num in self.FN])
        table.add_column('target_Precision',
                         ['{:.5f}'.format(num) for num in self.Precision])
        table.add_column('target_Recall',
                         ['{:.5f}'.format(num) for num in self.Recall])
        table.add_column('target_F1score',
                         ['{:.5f}'.format(num) for num in self.F1])
        print(table)

        return self.Precision, self.Recall, self.F1

    def reset(self):
        self.TP = np.zeros_like(self.dis_thrs)
        self.FP = np.zeros_like(self.dis_thrs)
        self.FN = np.zeros_like(self.dis_thrs)
        self.Precision = np.zeros_like(self.dis_thrs)
        self.Recall = np.zeros_like(self.dis_thrs)
        self.F1 = np.zeros_like(self.dis_thrs)

    @property
    def table(self):
        all_metric = np.stack([
            self.dis_thrs, self.TP, self.FP, self.FN, self.Precision,
            self.Recall, self.F1
        ],
                              axis=1)
        df = pd.DataFrame(all_metric)
        df.columns = ['Dis-Thr', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1']
        return df

    def _calculate_tp_fn_fp(self, distances: np.ndarray,
                            threshold: int) -> Tuple[int]:
        """_summary_

        Args:
            distances (np.ndarray): distances in shape (num_lbl * num_pred)
            threshold (int): threshold of distances.

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: TP, FN, FP
        """
        num_lbl, num_pred = distances.shape
        if num_lbl * num_pred == 0:
            # no lbl or no pred
            TP = 0

        elif self.match_alg == 'forloop':
            for i in range(num_lbl):
                for j in range(num_pred):
                    if distances[i, j] < threshold:
                        distances[:, j] = np.nan  # Set inf to mark matched
                        break
            TP = np.sum(np.isnan(distances)) // num_lbl

        else:
            raise ValueError(
                f"{self.match_alg} is not implemented, please use 'forloop' for match_alg."
            )

        FP = num_pred - TP
        FN = num_lbl - TP

        return TP, FN, FP

    def _calculate_precision_recall_f1(self):

        self.Precision = _safe_divide(self.TP, self.TP + self.FP)
        self.Recall = _safe_divide(self.TP, self.TP + self.FN)
        # micro F1 socre.
        self.F1 = _safe_divide(2 * self.Precision * self.Recall,
                               self.Precision + self.Recall)

    def __repr__(self) -> str:
        message = (f'{self.__class__.__name__}'
                   f'(match_alg={self.match_alg} '
                   f'conf_thr={self.conf_thr})')
        return message
