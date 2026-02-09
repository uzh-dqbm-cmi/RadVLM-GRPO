import re
from typing import List

import numpy as np


def extract_bounding_boxes(answer: str) -> List[List[float]]:
    """Extract [x1, y1, x2, y2] as floats; ignore malformed boxes.

    Supports negatives and scientific notation. Boxes with any non-finite
    or unparsable coordinate are skipped.
    """
    # strict numeric token: optional sign, decimals, optional exponent
    NUM = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
    pattern = rf"\[\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})\s*\]"

    boxes: List[List[float]] = []
    for m in re.finditer(pattern, answer):
        try:
            b = [float(m.group(1)), float(m.group(2)),
                 float(m.group(3)), float(m.group(4))]
        except Exception:
            # any unparsable number -> skip this bracketed group
            continue
        # drop non-finite coords
        if not all(np.isfinite(b)):
            continue
        boxes.append(b)
    return boxes



def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection over Union between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    denom = box1_area + box2_area - inter_area
    return inter_area / denom if denom > 0 else 0.0

from scipy.optimize import linear_sum_assignment

def compute_iou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    a2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))
    denom = a1 + a2 - inter
    return inter / denom if denom > 0 else 0.0

def _iou_matrix(predicted: List[List[float]], gt: List[List[float]]) -> np.ndarray:
    M = np.zeros((len(predicted), len(gt)), dtype=float)
    for i, p in enumerate(predicted):
        for j, g in enumerate(gt):
            M[i, j] = compute_iou(p, g)
    return M

def soft_f1_hungarian(
    predicted_boxes: List[List[float]],
    actual_boxes: List[List[float]],
    min_iou: float = 0.0,
) -> float:
    """
    IoU-weighted soft-F1 with *global* one-to-one (Hungarian) matching.

    TP_mass = sum of IoUs over matched pairs with IoU > min_iou
    FP_count = #predictions not part of a valid match
    Recall = TP_mass / (#GT)
    Precision = TP_mass / (TP_mass + FP_count)
    Soft-F1 = 2PR / (P+R)
    """
    n_pred, n_gt = len(predicted_boxes), len(actual_boxes)
    if n_pred == 0 or n_gt == 0:
        return 0.0

    ious = _iou_matrix(predicted_boxes, actual_boxes)

    # Maximize total IoU via Hungarian: minimize cost = 1 - IoU
    cost = 1.0 - ious
    row_ind, col_ind = linear_sum_assignment(cost)  # pairs of size min(n_pred, n_gt)

    matched_ious = ious[row_ind, col_ind]
    valid = matched_ious > min_iou  # keep only meaningful overlaps (> 0 by default)
    tp_mass = float(matched_ious[valid].sum())
    num_valid_matches = int(valid.sum())

    fp_count = n_pred - num_valid_matches
    gt_count = n_gt

    recall = tp_mass / (gt_count + 1e-8)
    precision = tp_mass / (tp_mass + fp_count + 1e-8)
    return float(2.0 * precision * recall / (precision + recall + 1e-8))


def compute_score(data_source, solution_str: str, ground_truth, extra_info) -> float:
    """Compute continuous soft-F1 reward between model output and ground-truth boxes.

    Rules:
    - Extract the answer ONLY from the text that follows a closing </think> tag.
    - If the closing </think> tag is missing, or nothing follows it, return 0.0.
    """

    #Find the FIRST closing </think> and capture everything after it (till end of string)

    if extra_info["thinking"]:
        m = re.search(r"</think>\s*(.*)\Z", solution_str or "", flags=re.I | re.S)
        if not m:
            return 0.0

        post_think_text = (m.group(1) or "").strip()
        if not post_think_text:
            return 0.0
    else:
        post_think_text = solution_str.strip()


    # Predicted boxes from text after </think>
    predicted_boxes = extract_bounding_boxes(post_think_text)

    # Ground-truth may be a list or a string; handle both
    if isinstance(ground_truth, str):
        ground_truth_boxes = extract_bounding_boxes(ground_truth)
    else:
        ground_truth_boxes = ground_truth or []

    return soft_f1_hungarian(predicted_boxes, ground_truth_boxes)
