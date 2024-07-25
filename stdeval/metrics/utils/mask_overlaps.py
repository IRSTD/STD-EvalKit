import numpy as np

from .misc import _safe_divide


def target_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """Compute Mask IoU between two groups of masks.

    Args:
        mask1 (np.ndarray): [num_gt, h, w] in one image, \
            num_gt is the number of gt in one image, use N to represent.
        mask2 (np.ndarray): [num_pred, h, w] in one image, \
            num_pred is the number of pred in one image, use M to represent.

    Returns:
        np.ndarray: NxM matrix, where N is the number of gt in mask1 and M is the number of pred in mask2.
    """
    mask1_area = np.count_nonzero(mask1 == 1, axis=(1, 2))[:, None]  # [N, 1]
    mask2_area = np.count_nonzero(mask2 == 1, axis=(1, 2))[None, ...]  # [1, M]
    intersection = np.count_nonzero(np.logical_and(mask1[:, None, ...] == 1,
                                                   mask2[None, ...] == 1),
                                    axis=(2, 3))  # [N, M, h, w] -> [N, M]
    iou = _safe_divide(intersection, mask1_area + mask2_area - intersection)
    return iou


def batch_binary_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """Compute batch binary mask iou.

    Args:
        mask1 (np.ndarray): [bs, h, w], bs is the batch size.
        mask2 (np.ndarray): [bs, h, w], bs is the batch size.

    Returns:
        np.ndarray: _description_
    """
    mask1_area = np.count_nonzero(mask1 == 1, axis=(1, 2))
    mask2_area = np.count_nonzero(mask2 == 1, axis=(1, 2))
    intersection = np.count_nonzero(np.logical_and(mask1 == 1, mask2 == 1),
                                    axis=(1, 2))
    iou = _safe_divide(intersection, (mask1_area + mask2_area - intersection))
    return iou
