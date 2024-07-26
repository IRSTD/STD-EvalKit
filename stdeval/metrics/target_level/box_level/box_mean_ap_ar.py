import warnings
from typing import Dict, List

import pandas as pd
import torch
from prettytable import PrettyTable
from torch import Tensor
from torchmetrics.detection import MeanAveragePrecision

from stdeval.metrics import BaseMetric, time_cost_deco


class BoxAveragePrecision(MeanAveragePrecision, BaseMetric):

    def __init__(self,
                 box_format='xyxy',
                 iou_type='bbox',
                 extended_summary: bool = False,
                 classwise: Dict[int, str] = {},
                 print_table=True,
                 **kwargs):
        """Compute the Mean-Average-Precision (mAP) and Mean-Average-Recall (mAR) \
            for object detection predictions(COCO).

        For ease to use, we encapsulate the MeanAveragePrecision of torchmetrics,
        and add some features to make it more user-friendly.
        1.We've added metrics and displays for each category.
        2.We've added a display of the all metric to ASCII table and DataFrame.
        theoretically supports all methods in torchmetrics.detection.mean_ap.MeanAveragePrecision.

        For more information, please refer to the official documentation:
        https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html#torchmetrics.detection.mean_ap.MeanAveragePrecision

        Usage:
            # For box:
                preds = [
                dict(
                    boxes=tensor([[258.0, 41.0, 606.0, 285.0],
                                [158.0, 41.0, 462.0, 285.0]]),
                    scores=tensor([0.536, 0.71]),
                    labels=tensor([1, 2]),
                ),
                    dict(
                    boxes=tensor([[254.0, 413.0, 656.0, 245.0]]),
                    scores=tensor([0.526]),
                    labels=tensor([1]),
                )
                ]
                target = [
                dict(
                    boxes=tensor([[214.0, 41.0, 562.0, 285.0],
                                [158.0, 41.0, 462.0, 285.0]]),
                    labels=tensor([1,2]),
                ),
                    dict(
                    boxes=tensor([[258.0, 41.0, 606.0, 285.0]]),
                    labels=tensor([1]),
                )
                ]
                classwise = {0:'person', 1:'car', 2:'tea', 3:'cycle'} # lbl id 2 name
                metric = BoxAveragePrecision(iou_type="bbox", class_metrics=True, classwise=classwise)
                metric.update(target, preds)
                metric.get()
                metric.table
                metric.reset()


            # For mask
                mask_pred = [
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0],
                    [0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    ]
                mask_tgt = [
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    ]
                preds = [
                    dict(
                        masks=tensor([mask_pred], dtype=torch.bool),
                        scores=tensor([0.536]),
                        labels=tensor([0]),
                    )
                    ]
                target = [
                    dict(
                        masks=tensor([mask_tgt], dtype=torch.bool),
                        labels=tensor([0]),
                    )
                    ]
                metric = BoxLevelMeanAveragePrecision(iou_type="segm")
                metric.update(target, preds)
                metric.get()

        Args:
            box_format (str, optional): Params of torchmetrics.detection.MeanAveragePrecision. \
                Defaults to 'xyxy'.
            iou_type (str, optional): Params of torchmetrics.detection.MeanAveragePrecision. \
                Defaults to "bbox".
            extended_summary (bool, optional): Params of torchmetrics.detection.MeanAveragePrecision. \
                Defaults to 'False'.
            classwise (Dict[int, str], optional): Methods function to this repo, controls whether use Category name, \
                like classwise = {0:'person', 1:'car', 2:'tea', 3:'cycle'}, label id map to name. Defaults to {}.
            print_table (bool, optional): Methods specific to this repo, controls whether an ASCII table to printed. \
                Defaults to True.
            **kwargs: Other keyword arguments for torchmetrics.detection.MeanAveragePrecision.
        """
        self.this_extend_summary = extended_summary
        self.classwise = classwise

        MeanAveragePrecision.__init__(self,
                                      iou_type=iou_type,
                                      box_format=box_format,
                                      extended_summary=True,
                                      **kwargs)
        BaseMetric.__init__(self, print_table=print_table)
        if len(self.iou_type) == 1:
            self.prefix = ''
        else:
            raise ValueError('not support iou_type with multiple values')

        iou_thr = str(round(self.iou_thresholds[0], 2)) + ':' + str(
            round(self.iou_thresholds[-1], 2))
        self.name2coco = {
            f'mAP@{iou_thr}':
            'map',
            'mAP@50':
            f'{self.prefix}map_50',
            'mAP@75':
            f'{self.prefix}map_75',
            'mAP_s':
            f'{self.prefix}map_small',
            'mAP_m':
            f'{self.prefix}map_medium',
            'mAP_l':
            f'{self.prefix}map_large',
            'mAR_s':
            f'{self.prefix}mar_small',
            'mAR_m':
            f'{self.prefix}mar_medium',
            'mAR_l':
            f'{self.prefix}mar_large',
            f'mAR_max_dets@{self.max_detection_thresholds[0]}':
            f'{self.prefix}mar_{self.max_detection_thresholds[0]}',
            f'mAR_max_dets@{self.max_detection_thresholds[1]}':
            f'{self.prefix}mar_{self.max_detection_thresholds[1]}',
            f'mAR_max_dets@{self.max_detection_thresholds[2]}':
            f'{self.prefix}mar_{self.max_detection_thresholds[2]}',
        }

    @time_cost_deco
    def update(self, labels: List[Dict[str, Tensor]],
               preds: List[Dict[str, Tensor]]) -> None:
        MeanAveragePrecision.update(self, preds, labels)

    @time_cost_deco
    def get(self) -> dict:
        res = MeanAveragePrecision.compute(self)
        results_per_category = self._get_per_class_info(res)
        results = dict()

        if not self.this_extend_summary:
            del res[f'{self.prefix}precision']
            del res[f'{self.prefix}recall']
            del res[f'{self.prefix}scores']
            del res[f'{self.prefix}ious']
            results['classes'] = res.pop('classes')
            results['ALL'] = res
            results.update(results_per_category)
        else:
            results[f'{self.prefix}precision'] = res.pop(
                f'{self.prefix}precision')
            results[f'{self.prefix}recall'] = res.pop(f'{self.prefix}recall')
            results[f'{self.prefix}scores'] = res.pop(f'{self.prefix}scores')
            results[f'{self.prefix}ious'] = res.pop(f'{self.prefix}ious')
            results['classes'] = res.pop('classes')
            results['ALL'] = res
            results.update(results_per_category)

        self.results = results
        if self.print_table:
            table = PrettyTable()
            head = ['category']
            head.extend([k for k, v in self.name2coco.items()])
            table.field_names = head
            all_row = ['All']
            all_row.extend([
                f"{self.results['ALL'][v].item():.4f}"
                for k, v in self.name2coco.items()
            ])
            table.add_row(all_row)
            for cls_idx in self.results['classes'].tolist():
                row = [self.results[cls_idx]['name']]
                row.extend([
                    f'{self.results[cls_idx][v].item():.4f}'
                    for k, v in self.name2coco.items()
                ])
                table.add_row(row)
            print(table)
        return self.results

    def reset(self):
        self.results = dict()
        MeanAveragePrecision.reset(self)

    @property
    def table(self):
        head = ['category']
        head.extend([k for k, v in self.name2coco.items()])
        all_row = ['All']
        data = []

        all_row.extend([
            f"{self.results['ALL'][v].item():.4f}"
            for k, v in self.name2coco.items()
        ])
        data.append(all_row)
        for cls_idx in self.results['classes'].tolist():
            row = [self.results[cls_idx]['name']]
            row.extend([
                f'{self.results[cls_idx][v].item():.4f}'
                for k, v in self.name2coco.items()
            ])
            data.append(row)
        df = pd.DataFrame(data).T
        df.index = head
        return df.T

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(iou_type={self.iou_type}, '
            f'box_format={self.box_format}, '
            f'iou_threshold={round(self.iou_thresholds[0],2)}:{round(self.iou_thresholds[-1],2)}, '
            f'rec_threshold={round(self.rec_thresholds[0],2)}:{round(self.rec_thresholds[-1],2)})'
        )

    def _get_per_class_info(self, results: dict):
        cat_ids2name = self.classwise
        precisions = results['precision']
        recalls = results['recall']
        classes = results['classes']
        iou_thrs = torch.tensor(self.iou_thresholds)
        max_dets = self.max_detection_thresholds
        # assert len(cat_ids) == precisions.shape[2]
        results_per_category = {}
        num_iou_thr, num_cls, num_area_rng, num_max_dets = recalls.shape

        num_iou_thr, num_rec_thr, num_cls, num_area_rng, num_max_dets = precisions.shape

        if len(cat_ids2name) != 0:
            for idx, cls_name in cat_ids2name.items():
                if not sum(classes[classes == idx]):
                    warnings.warn(
                        f"Category '{cls_name}', ID = {idx} not in preds or targets, but in classwise."
                        f'This information can be ignored if you are sure there is not a problem.'
                    )

        for idx, cls in enumerate(classes):
            cls = cls.item()
            if len(cat_ids2name) != 0:
                assert cls in cat_ids2name.keys(
                ), f'lables not in catIds2name, {cls} not in {cat_ids2name.keys()}'
                cat_name = str(cat_ids2name[cls])
            else:
                cat_name = str(cls)

            t = dict(name=cat_name)
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            if precision.numel():
                ap = torch.mean(precision)
            else:
                ap = torch.tensor([-1])
            t[f'{self.prefix}map'] = ap

            # get .5,, .75.
            iou_idx = [torch.where(iou_thrs == iou)[0] for iou in [0.5, 0.75]]
            ap = []
            for iou, iou_t, in zip(iou_idx, [0.5, 0.75]):
                precision = precisions[iou, :, idx, 0, -1]
                precision = precision[precision > -1]
                if precision.numel():
                    ap.append(torch.mean(precision))
                else:
                    ap.append(torch.tensor([-1]))

            t[f'{self.prefix}map_50'] = ap[0]
            t[f'{self.prefix}map_75'] = ap[1]

            # get small, medium, large
            ap = []
            for area in range(1, num_area_rng):  # 1,2,3; 0 is all area
                precision = precisions[:, :, idx, area, -1]
                precision = precision[precision > -1]
                if precision.numel():
                    ap.append(torch.mean(precision))
                else:
                    ap.append(torch.tensor([-1]))

            t[f'{self.prefix}map_small'] = ap[0]
            t[f'{self.prefix}map_medium'] = ap[1]
            t[f'{self.prefix}map_large'] = ap[2]

            # mAR
            recall = recalls[:, idx, 0, -1]
            recall = recall[recall > -1]
            if recall.numel():
                ar = torch.mean(recall)
            else:
                ar = torch.tensor([-1])
            t[f'{self.prefix}mar'] = ar

            # small medium large
            ar = []
            for area in range(1, num_area_rng):  # 1,2,3; 0 is all area
                recall = recalls[:, idx, area, -1]
                recall = recall[recall > -1]
                if recall.numel():
                    ar.append(torch.mean(recall))
                else:
                    ar.append(torch.tensor([-1]))
            t[f'{self.prefix}mar_small'] = ar[0]
            t[f'{self.prefix}mar_medium'] = ar[1]
            t[f'{self.prefix}mar_large'] = ar[2]

            for jdx, max_det in enumerate(max_dets):  # 1,2,3; 0 is all area
                recall = recalls[:, idx, 0, jdx]
                recall = recall[recall > -1]
                if recall.numel():
                    ar = torch.mean(recall)
                else:
                    ar = torch.tensor([-1])
                t[f'{self.prefix}mar_{max_det}'] = ar

            results_per_category[cls] = t
        return results_per_category
