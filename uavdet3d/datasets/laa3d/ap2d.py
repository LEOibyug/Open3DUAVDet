import numpy as np
from typing import List, Tuple, Dict, Any


def calculate_ap_single_threshold(pred_boxes: np.ndarray,
                                  pred_scores: np.ndarray,
                                  pred_frame_ids: np.ndarray,
                                  gt_boxes: np.ndarray,
                                  gt_frame_ids: np.ndarray,
                                  iou_threshold: float,
                                  recall_number: int = 41) -> tuple:
    """
    Calculate AP for a single IoU threshold.

    Args:
        pred_boxes: Predicted boxes (N, 4)
        pred_scores: Prediction confidence scores (N,)
        pred_frame_ids: Frame IDs for predictions (N,)
        gt_boxes: Ground truth boxes (M, 4)
        gt_frame_ids: Frame IDs for ground truth (M,)
        iou_threshold: IoU threshold for matching
        recall_number: Number of recall points for interpolation

    Returns:
        Tuple of (Average Precision value, interpolated precision array)
    """

    # Edge case 1: No ground truth boxes
    if len(gt_boxes) == 0:
        if len(pred_boxes) == 0:
            # Both empty - perfect match
            return 1.0, np.ones(recall_number)
        else:
            # Predictions but no ground truth - all false positives
            return 0.0, np.zeros(recall_number)

    # Edge case 2: No predictions but have ground truth
    if len(pred_boxes) == 0:
        return 0.0, np.zeros(recall_number)

    # Validate input shapes
    assert pred_boxes.shape[0] == pred_scores.shape[0] == pred_frame_ids.shape[0], \
        "Prediction arrays must have same length"
    assert gt_boxes.shape[0] == gt_frame_ids.shape[0], \
        "Ground truth arrays must have same length"
    assert pred_boxes.shape[1] == gt_boxes.shape[1] == 4, \
        "Boxes must have 4 coordinates"

    # Sort predictions by confidence score (descending)
    # Use stable sort with original indices as secondary key to ensure deterministic behavior
    original_indices = np.arange(len(pred_scores))
    # Sort by: (-score, original_index) to ensure stable, deterministic sorting
    sorted_indices = np.lexsort((original_indices, -pred_scores))

    pred_boxes_sorted = pred_boxes[sorted_indices]
    pred_scores_sorted = pred_scores[sorted_indices]
    pred_frame_ids_sorted = pred_frame_ids[sorted_indices]

    # Get unique frame IDs
    unique_frames = np.unique(np.concatenate([pred_frame_ids, gt_frame_ids]))

    # Initialize tracking arrays
    num_preds = len(pred_boxes_sorted)
    num_gts = len(gt_boxes)

    tp = np.zeros(num_preds)  # True positives
    fp = np.zeros(num_preds)  # False positives
    gt_matched = {}  # Track which GT boxes have been matched

    # Initialize GT matched status for each frame
    for frame_id in unique_frames:
        frame_gt_mask = gt_frame_ids == frame_id
        num_gt_in_frame = np.sum(frame_gt_mask)
        if num_gt_in_frame > 0:
            gt_matched[frame_id] = np.zeros(num_gt_in_frame, dtype=bool)

    # Process each prediction
    for pred_idx in range(num_preds):
        pred_box = pred_boxes_sorted[pred_idx]
        pred_frame = pred_frame_ids_sorted[pred_idx]

        # Get ground truth boxes in the same frame
        gt_frame_mask = gt_frame_ids == pred_frame
        gt_boxes_in_frame = gt_boxes[gt_frame_mask]

        if len(gt_boxes_in_frame) == 0:
            # No GT in this frame - false positive
            fp[pred_idx] = 1
            continue

        # Calculate IoU with all GT boxes in the frame
        #
        # print(pred_box)
        # print(gt_boxes_in_frame)


        ious = calculate_iou(pred_box, gt_boxes_in_frame)

        # Find best matching GT box
        max_iou_idx = np.argmax(ious)
        max_iou = ious[max_iou_idx]

        # print(ious)
        # input()
        # Check if IoU meets threshold and GT not already matched
        if max_iou >= iou_threshold:
            if not gt_matched[pred_frame][max_iou_idx]:
                # True positive
                tp[pred_idx] = 1
                gt_matched[pred_frame][max_iou_idx] = True
            else:
                # GT already matched - false positive
                fp[pred_idx] = 1
        else:
            # IoU below threshold - false positive
            fp[pred_idx] = 1

    # Compute cumulative TP and FP
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    # Compute precision and recall
    recall = tp_cumsum / num_gts
    precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(float).eps)

    # Handle edge case where all predictions are correct and all GTs are matched
    if len(recall) > 0 and np.isclose(recall[-1], 1.0) and np.isclose(precision[-1], 1.0):
        # Perfect predictions - all GTs matched with no false positives
        return 1.0, np.ones(recall_number)

    # Interpolate precision at recall_number points
    recall_points = np.linspace(0, 1, recall_number)
    interpolated_precision = np.zeros(recall_number)

    # Add boundary points
    recall = np.concatenate([[0], recall])

    precision = np.concatenate([[1 if len(tp_cumsum) > 0 and tp_cumsum[0] > 0 else 0], precision])
    # print(precision)
    # print(recall)
    # input()
    # Compute maximum precision for all recalls >= recall_points[i] (monotonically decreasing envelope)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # Interpolate precision values
    for i, r in enumerate(recall_points):
        # Find the first recall >= r
        indices = np.where(recall >= r)[0]
        if len(indices) > 0:
            interpolated_precision[i] = precision[indices[0]]
        else:
            # No recall point >= r, use 0
            interpolated_precision[i] = 0

    # Calculate AP as mean of interpolated precision
    ap = np.mean(interpolated_precision)

    return ap, interpolated_precision


def calculate_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Calculate IoU (including degenerate cases such as lines or points)
    between one box and multiple boxes.

    Args:
        box: Single box (4,) in format [x1, y1, x2, y2]
        boxes: Multiple boxes (N, 4) in format [x1, y1, x2, y2]

    Returns:
        IoU values (N,)
    """
    # Normalize coordinates to ensure x1 < x2, y1 < y2
    box = np.array([
        min(box[0], box[2]),
        min(box[1], box[3]),
        max(box[0], box[2]),
        max(box[1], box[3])
    ], dtype=float)
    boxes = boxes.copy().astype(float)
    boxes[:, [0, 2]] = np.sort(boxes[:, [0, 2]], axis=1)
    boxes[:, [1, 3]] = np.sort(boxes[:, [1, 3]], axis=1)

    # Compute box widths/heights
    box_w, box_h = box[2] - box[0], box[3] - box[1]
    boxes_w = boxes[:, 2] - boxes[:, 0]
    boxes_h = boxes[:, 3] - boxes[:, 1]

    # Intersection
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection_w = np.maximum(0.0, x2 - x1)
    intersection_h = np.maximum(0.0, y2 - y1)

    intersection_area = intersection_w * intersection_h

    # Union
    box_area = box_w * box_h
    boxes_area = boxes_w * boxes_h
    union_area = box_area + boxes_area - intersection_area

    iou = np.zeros_like(intersection_area, dtype=float)

    # --- Case 1: 正常矩形 ---
    normal_mask = (box_w > 0) & (box_h > 0) & (boxes_w > 0) & (boxes_h > 0)
    iou[normal_mask] = intersection_area[normal_mask] / np.maximum(union_area[normal_mask], 1e-12)

    # --- Case 2: 退化为水平线 ---
    horiz_mask = (box_h == 0) & (boxes_h == 0)
    if np.any(horiz_mask):
        inter_len = np.maximum(0.0, np.minimum(box[2], boxes[horiz_mask, 2]) - np.maximum(box[0], boxes[horiz_mask, 0]))
        union_len = (box_w + boxes_w[horiz_mask] - inter_len)
        iou[horiz_mask] = np.where(union_len > 0, inter_len / union_len, 0.0)

    # --- Case 3: 退化为垂直线 ---
    vert_mask = (box_w == 0) & (boxes_w == 0)
    if np.any(vert_mask):
        inter_len = np.maximum(0.0, np.minimum(box[3], boxes[vert_mask, 3]) - np.maximum(box[1], boxes[vert_mask, 1]))
        union_len = (box_h + boxes_h[vert_mask] - inter_len)
        iou[vert_mask] = np.where(union_len > 0, inter_len / union_len, 0.0)

    # --- Case 4: 点（宽高都为0） ---
    point_mask = (box_w == 0) & (box_h == 0) & (boxes_w == 0) & (boxes_h == 0)
    if np.any(point_mask):
        same_point = (np.abs(boxes[point_mask, 0] - box[0]) < 1e-8) & (np.abs(boxes[point_mask, 1] - box[1]) < 1e-8)
        iou[point_mask] = same_point.astype(float)  # 相同点IoU=1，否则0

    # --- Case 5: 完全一致框（精度修正） ---
    same_mask = np.allclose(boxes, box, atol=1e-8)
    iou[same_mask] = 1.0

    # Clip to [0, 1]
    iou = np.clip(iou, 0.0, 1.0)
    return iou


def calculate_object_detection_2dap(pred_boxes: List[List[float]],
                                    pred_scores: List[float],
                                    pred_frame_ids: List[int],
                                    gt_boxes: List[List[float]],
                                    gt_frame_ids: List[int],
                                    matching_thresholds: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                    recall_number: int = 41) -> Dict[str, Any]:
    """
    Calculate object detection Average Precision (AP) with IoU-based matching.

    Args:
        pred_boxes: List of predicted bounding boxes, each in format [x1, y1, x2, y2]
        pred_scores: List of prediction confidence scores
        pred_frame_ids: List of frame IDs for predictions
        gt_boxes: List of ground truth bounding boxes, each in format [x1, y1, x2, y2]
        gt_frame_ids: List of frame IDs for ground truth
        matching_thresholds: List of IoU thresholds for matching
        recall_number: Number of recall points for interpolation

    Returns:
        Dictionary containing AP results
    """
    # Convert inputs to numpy arrays
    pred_boxes = np.array(pred_boxes, dtype=float)
    pred_scores = np.array(pred_scores, dtype=float)
    pred_frame_ids = np.array(pred_frame_ids, dtype=int)
    gt_boxes = np.array(gt_boxes, dtype=float)
    gt_frame_ids = np.array(gt_frame_ids, dtype=int)

    # Handle empty cases
    if len(pred_boxes) == 0:
        pred_boxes = pred_boxes.reshape(0, 4)
    if len(gt_boxes) == 0:
        gt_boxes = gt_boxes.reshape(0, 4)

    # Validate inputs
    if len(pred_boxes) > 0:
        assert len(pred_boxes) == len(pred_scores) == len(pred_frame_ids), \
            "Prediction arrays must have same length"
        assert pred_boxes.shape[1] == 4, "Bounding boxes must have 4 coordinates"

    if len(gt_boxes) > 0:
        assert len(gt_boxes) == len(gt_frame_ids), \
            "Ground truth arrays must have same length"
        assert gt_boxes.shape[1] == 4, "Bounding boxes must have 4 coordinates"

    # Calculate AP for each IoU threshold
    ap_results = {}
    ap_values = []
    detailed_ap_values = []

    for threshold in matching_thresholds:
        ap, all_values = calculate_ap_single_threshold(
            pred_boxes, pred_scores, pred_frame_ids,
            gt_boxes, gt_frame_ids, threshold, recall_number
        )

        ap_results[f'AP@{threshold:.1f}'] = ap
        ap_values.append(ap)
        detailed_ap_values.append(all_values)

    # Calculate mean AP across all thresholds
    mean_ap = np.mean(ap_values)

    return mean_ap, ap_values, detailed_ap_values