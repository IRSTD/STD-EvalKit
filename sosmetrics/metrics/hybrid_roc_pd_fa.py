import threading
from typing import Any, List, Union

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn.metrics import auc

from .base import time_cost_deco
from .hybrid_pd_fa import TargetPdPixelFa
from .utils import (_TYPES, _adjust_conf_thr_arg, _safe_divide,
                    calculate_target_infos, convert2format,
                    get_label_coord_and_gray, get_pred_coord_and_gray,
                    second_match_method)


class TargetPdPixelFaROC(TargetPdPixelFa):

    def __init__(self,
                 conf_thrs: Union[int, List[float], np.ndarray] = 10,
                 dis_thrs: Union[List[int], int] = [1, 10],
                 match_alg: str = 'forloop',
                 second_match: str = 'none',
                 **kwargs: Any):
        """Calculation of ROC using TargetPdPixelFa.
            More details can be found at sosmetrics.metrics.TargetPdPixelFa.

        Args:
            conf_thrs (Union[int, List[float], np.ndarray]): Confidence thresholds.
                - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced \
                    from 0 to 1 as conf_thrs for the calculation.
                - If set to an `list` of floats, will use the indicated thresholds in the list as conf_thrs \
                    for the calculation
                - If set to an 1d `array` of floats, will use the indicated thresholds in the array as
                    conf_thrs for the calculation.
            Other parameters are same as TargetPdPixelFa.
        """
        self.conf_thrs = _adjust_conf_thr_arg(conf_thrs)
        super().__init__(dis_thrs=dis_thrs,
                         conf_thr=0.5,
                         match_alg=match_alg,
                         second_match=second_match,
                         **kwargs)
        self.lock = threading.Lock()
        self.reset()

    @time_cost_deco
    def update(self, labels: _TYPES, preds: _TYPES) -> None:

        def evaluate_worker(self, label: np.array, pred: np.array) -> None:
            # to unit8 for ``convert2gray()``
            coord_label, gray_label = get_label_coord_and_gray(label)

            for idx, conf_thr in enumerate(self.conf_thrs):
                coord_pred, gray_pred = get_pred_coord_and_gray(
                    pred.copy(), conf_thr)
                distances, mask_iou, bbox_iou = calculate_target_infos(
                    coord_label, coord_pred, gray_pred.shape[0],
                    gray_pred.shape[1])

                if self.debug:
                    print(f'bbox_iou={bbox_iou}')
                    print(f'mask_iou={mask_iou}')
                    print(f'eul_distance={distances}')
                    print('____' * 20)

                if self.second_match != 'none':
                    distances = second_match_method(distances, mask_iou,
                                                    bbox_iou,
                                                    self.second_match)
                    if self.debug:
                        print(f'After second match eul distances={distances}')
                        print('____' * 20)

                for jdx, threshold in enumerate(self.dis_thrs):
                    AT, TD, FD, NP = self._calculate_at_td_fd_np(
                        distances.copy(), coord_pred, threshold, gray_pred)
                    with self.lock:
                        self.AT[jdx, idx] += AT
                        self.FD[jdx, idx] += FD
                        self.NP[jdx, idx] += NP
                        self.TD[jdx, idx] += TD

        # Packaged in the format we need, bhwc of np.array or hwc of list.
        labels, preds = convert2format(labels, preds)

        # for i in range(len(labels)):
        #     evaluate_worker(labels[i].squeeze(0), preds[i].squeeze(0))
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

        self.FA = _safe_divide(self.FD, self.NP)
        self.PD = _safe_divide(self.TD, self.AT)

        index = np.argsort(self.FA, axis=-1)
        fa = np.take_along_axis(self.FA, index, axis=1)
        pd = np.take_along_axis(self.PD, index, axis=1)
        fa = np.concatenate([np.zeros((fa.shape[0], 1)), fa], axis=-1)
        pd = np.concatenate([np.ones((pd.shape[0], 1)), pd], axis=1)

        self.auc = [auc(fa[i], pd[i]) for i in range(len(self.dis_thrs))]

        if self.print_table:
            head = ['Dis—Thr']
            head.extend(self.dis_thrs.tolist())
            table = PrettyTable()
            table.field_names = head
            auc_row = ['AUC-ROC']
            auc_row.extend(['{:.4f}'.format(num) for num in self.auc])
            table.add_row(auc_row)

            print(table)

        return self.PD, self.FA, self.auc

    def reset(self) -> None:
        self.FA = np.zeros((len(self.dis_thrs), len(self.conf_thrs)))
        self.TD = np.zeros((len(self.dis_thrs), len(self.conf_thrs)))
        self.FD = np.zeros((len(self.dis_thrs), len(self.conf_thrs)))
        self.NP = np.zeros((len(self.dis_thrs), len(self.conf_thrs)))
        self.AT = np.zeros((len(self.dis_thrs), len(self.conf_thrs)))
        self.PD = np.zeros(len(self.dis_thrs))
        self.auc = np.zeros(len(self.dis_thrs))

    @property
    def table(self):
        all_metric = np.vstack([self.dis_thrs, self.auc])
        df = pd.DataFrame(all_metric)
        df.index = ['Dis-Thr', 'AUC-ROC']
        return df

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(match_alg={self.match_alg})'
