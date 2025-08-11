import torch

def convert_to_tracker_format(boxes, labels, confidence=0.9):
    """
    Convert bounding boxes and labels to tracker format:
    ([left, top, width, height], confidence, detection_class)

    Args:
        boxes (torch.Tensor): shape [N, 4], in [x1, y1, x2, y2] format
        labels (torch.Tensor): shape [N], integer class labels
        confidence (float): confidence score for all boxes

    Returns:
        List[Tuple[List[float], float, int]]
    """
    # if boxes.is_cuda:
    # print(type(boxes))
    boxes = boxes.tensor.cpu().numpy()
    # print(boxes)
    # if labels.is_cuda:
    labels = labels.cpu().numpy()
    # print(labels)

    detections = []
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box.tolist()
        left = x1
        top = y1
        width = x2 - x1
        height = y2 - y1
        det = ([left, top, width, height], confidence, int(label))
        detections.append(det)

    return detections


# Example usage:
# if __name__ == "__main__":
#     boxes = torch.tensor([
#         [785.5082, 385.8770, 1084.0192, 510.8224],
#         [270.3557, 384.6895, 502.0558, 496.8985]
#     ], device='cuda:0')

#     labels = torch.tensor([2, 2], device='cuda:0')

#     detections = convert_to_tracker_format(boxes, labels, confidence=0.9)

#     print(detections)
# from bondinboxesConverter import convert_to_tracker_format
