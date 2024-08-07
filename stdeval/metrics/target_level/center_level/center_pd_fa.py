import threading
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from skimage.measure._regionprops import RegionProperties

from stdeval.metrics.base import BaseMetric, time_cost_deco
from stdeval.metrics.utils import (_TYPES, _adjust_dis_thr_arg, _safe_divide,
                                   calculate_target_infos, convert2format,
                                   get_label_coord_and_gray,
                                   get_pred_coord_and_gray,
                                   second_match_method)


class CenterPdPixelFa(BaseMetric):

    def __init__(self,
                 conf_thr: float = 0.5,
                 dis_thrs: Union[List[int], int] = [1, 10],
                 match_alg: str = 'forloop',
                 second_match: str = 'none',
                 **kwargs: Any):
        """
        Center Level Pd and Pixel Level Fa.
        Original Code: https://github.com/XinyiYing/BasicIRSTD/blob/main/metrics.py

        Paper:
            @ARTICLE{9864119,
            author={Li, Boyang and Xiao, Chao and Wang, Longguang and Wang, Yingqian and Lin, \
                Zaiping and Li, Miao and An, Wei and Guo, Yulan},
            journal={IEEE Transactions on Image Processing},
            title={Dense Nested Attention Network for Infrared Small Center Detection},
            year={2023},
            volume={32},
            number={},
            pages={1745-1758},
            keywords={Feature extraction;Object detection;Shape;Clutter;Decoding;Annotations;\
                Training;Infrared small target detection;deep learning;dense nested interactive module;\
                    channel and spatial attention;dataset},
            doi={10.1109/TIP.2022.3199107}}

        We have made the following improvements
            1. Supports multi-threading as well as batch processing.
            2. Supports secondary matching using mask iou.

        Original setting in above Paper:
            CenterPdPixelFa(
                        conf_thr=0.5,
                        dis_thrs=[1,10],
                        match_alg='forloop',
                        second_match='none'
                        )

        Pipeline:
            1. Get connectivity region of gt and pred.
            2. Iteration computes the Euclidean distance between the centroid of the connected regions of pred and gt.
                if distance < threshold, then match.
            3. Compute the following elements

                TD: Number of correctly predicted targets, \
                    GT is positive and Pred is positive, like TP.
                AT: All Centers, Number of target in GT, like TP + FN.
                PD: Probability of Detection, PD =TD/AT, like Recall = TP/(TP+FN).
                NP: All image Pixels, NP = H*W*num_gt_img.
                FD: The numbers of falsely predicted pixels, dismatch pixel, \
                    FD = NP - pixel_of_each_TD = FP * pixel_of_each_FP.
                FA: False-Alarm Rate, FA = FD/NP.

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

        def evaluate_worker(self, label: np.array, pred: np.array) -> None:
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
                AT, TD, FD, NP = self._calculate_at_td_fd_np(
                    distances.copy(), coord_pred, threshold, gray_pred)
                with self.lock:
                    self.AT[idx] += AT
                    self.FD[idx] += FD
                    self.NP[idx] += NP
                    self.TD[idx] += TD

        # Packaged in the format we need, bhwc of np.array or hwc of list.
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

        self.FA = _safe_divide(self.FD, self.NP)
        self.PD = _safe_divide(self.TD, self.AT)

        if self.print_table:
            head = ['Threshold']
            head.extend(self.dis_thrs.tolist())
            table = PrettyTable()
            table.add_column('Threshold', self.dis_thrs)
            table.add_column('TD', ['{:.0f}'.format(num) for num in self.TD])
            table.add_column('AT', ['{:.0f}'.format(num) for num in self.AT])
            table.add_column('FD', ['{:.0f}'.format(num) for num in self.FD])
            table.add_column('NP', ['{:.0f}'.format(num) for num in self.NP])
            table.add_column('target_Pd',
                             ['{:.5f}'.format(num) for num in self.PD])
            table.add_column('pixel_Fa',
                             ['{:.5e}'.format(num) for num in self.FA])
            print(table)

        return self.PD, self.FA

    def reset(self) -> None:
        self.FA = np.zeros_like(self.dis_thrs)
        self.TD = np.zeros_like(self.dis_thrs)
        self.FD = np.zeros_like(self.dis_thrs)
        self.NP = np.zeros_like(self.dis_thrs)
        self.AT = np.zeros_like(self.dis_thrs)
        self.PD = np.zeros_like(self.dis_thrs)

    @property
    def table(self):
        all_metric = np.stack([
            self.dis_thrs, self.TD, self.AT, self.FD, self.NP, self.PD, self.FA
        ],
                              axis=1)
        df_pd_fa = pd.DataFrame(all_metric)
        df_pd_fa.columns = [
            'Dis_thr', 'TD', 'AT', 'FD', 'NP', 'target_Pd', 'pixel_Fa'
        ]
        return df_pd_fa

    def _calculate_at_td_fd_np(self, distances: np.ndarray,
                               coord_pred: List[RegionProperties],
                               threshold: int, pred_img: np.ndarray) -> Tuple:
        """_summary_

        Args:
            distances (np.array): distances in shape (num_lbl * num_pred)
            coord_pred (List[RegionProperties]): measure.regionprops(pred)
            threshold (int): _description_
            pred_img (np.array): _description_

        Returns:
            tuple[int, int, int, int]: AT, TD, FD, NP
        """
        num_lbl, num_pred = distances.shape
        true_img = np.zeros(pred_img.shape)

        if num_lbl * num_pred == 0:
            # no lbl or no pred
            TD = 0

        elif self.match_alg == 'forloop':
            for i in range(num_lbl):
                for j in range(num_pred):
                    if distances[i, j] < threshold:
                        distances[:,
                                  j] = np.nan  # Set inf to mark matched preds.
                        true_img[coord_pred[j].coords[:, 0],
                                 coord_pred[j].coords[:, 1]] = 1
                        break
            # get number of inf columns, is equal to TD
            TD = np.sum(np.isnan(distances)) // num_lbl

        else:
            raise ValueError(f'Unknown match_alg: {self.match_alg}')

        FD = (pred_img - true_img).sum()
        NP = pred_img.shape[0] * pred_img.shape[1]
        AT = num_lbl
        return AT, TD, FD, NP

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(conf_thr={self.conf_thr}, '
                f'match_alg={self.match_alg}, '
                f'second_match={self.second_match})')
