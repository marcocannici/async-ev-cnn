import numpy as np


def center_crop(l, x, y, ts, p, bboxes, old_shape, new_shape):
    """
    Crops events and annotations to a centered region of the specified shape.
    Events and bounding boxes are then shifted so that the top-left event margins
    always start at (0,0)
    """

    new_h, new_w = new_shape
    old_h, old_w = old_shape

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    new_top = (x_max - x_min - new_w) // 2
    new_left = (y_max - y_min - new_h) // 2

    events_inside = np.logical_and.reduce([x >= new_left, x < new_left + new_w,
                                           y >= new_top, y < new_top + new_h])
    new_x, new_y, new_ts, new_p = x[events_inside], y[events_inside], \
                                  ts[events_inside], p[events_inside]
    new_x -= new_x.min()
    new_y -= new_y.min()
    new_l = new_x.shape[0]

    bboxes[:, [0, 2]] *= old_w
    bboxes[:, [1, 3]] *= old_h

    new_bboxes = bboxes.copy()
    new_bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]] * old_w - new_x.min(), 0, new_w) / new_w
    new_bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]] * old_h - new_x.min(), 0, new_h) / new_h

    return new_l, new_x, new_y, new_ts, new_p, new_bboxes


def apply_nms(batch_bboxes, batch_scores, batch_valid=None, iou_threshold=0.5):
    """
    Applies Non-Maximum-Suppression on the provided boxes.
    Implementation taken from: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

    :param batch_bboxes: a [batch_size, num_boxes, 4] array providing the parameters of the bounding boxes
        (x_center, y_center, w_box, h_box).
    :param batch_scores: a [batch_size, num_boxes] array providing the scores associated with each bounding box
    :param batch_valid: a [batch_size, num_boxes] boolean mask used to specify which values must be considered valid
        across the batch. Optional, if not provided, all the boxes will be considered in the computation
    :param iou_threshold: scalar, the threshold on the IOU. Optional, default 0.5.
    :return: a list of 2 numpy arrays representing the indices in batch_bboxes of the selected boxes
    """

    batch_valid = batch_valid if batch_valid is not None else [None] * batch_bboxes.shape[0]

    picked_idx = []
    # Loops over the batch dimension
    for bboxes, scores, valid in zip(batch_bboxes, batch_scores, batch_valid):

        if valid is not None:
            bboxes = bboxes[valid]
            scores = scores[valid]
            # compute mapping from valid indices to original
            valid_idx_to_original = np.where(valid)[0]

        # if there are no boxes
        if len(bboxes) == 0:
            picked_idx.append([])
        else:
            # initialize the list of picked indexes
            pick = []

            # grab the coordinates of the bounding boxes
            x = bboxes[:, 0]
            y = bboxes[:, 1]
            w = bboxes[:, 2]
            h = bboxes[:, 3]

            # compute the area of the bounding boxes and sort the bounding
            # boxes by their score
            area = w * h
            idxs = np.argsort(scores)

            # keep looping while some indexes still remain in the indexes
            # list
            while len(idxs) > 0:
                # grab the last index in the indexes list and add the
                # index value to the list of picked indexes
                last = len(idxs) - 1
                i = idxs[last]
                pick.append(i)

                # compute the top-left and bottom-right coordinates of the intersections
                # between the current box and all the remaining ones
                xx1 = np.maximum(x[i] - w[i] / 2, x[idxs[:last]] - w[idxs[:last]] / 2)
                yy1 = np.maximum(y[i] - h[i] / 2, y[idxs[:last]] - h[idxs[:last]] / 2)
                xx2 = np.minimum(x[i] + w[i] / 2, x[idxs[:last]] + w[idxs[:last]] / 2)
                yy2 = np.minimum(y[i] + h[i] / 2, y[idxs[:last]] + h[idxs[:last]] / 2)

                # compute the width and height of the intersection's boxes
                ww = np.maximum(0, xx2 - xx1)
                hh = np.maximum(0, yy2 - yy1)

                # compute IOUs
                iou = (ww * hh) / (area[idxs[:last]] + area[i] - (ww * hh))

                # delete from the list of remaining indexes, the current one (last) and those
                # of the bounding boxes with an IOU above the threshold with the current box
                idxs = np.delete(idxs, np.concatenate(([last], np.where(iou >= iou_threshold)[0])))

            # if a 'batch_valid' array has been provided, 'pick' will contain the indices of the filtered
            # boxes, we need to map them back to original array's indices
            pick = pick if valid is None else list(valid_idx_to_original[pick])
            picked_idx.append(pick)

    # HACK: it uses sum(list of lists, []) to flatten the list  =D
    idx_axis_0 = np.array(sum([[batch] * len(idx) for batch, idx in enumerate(picked_idx)], []))
    idx_axis_1 = np.array(sum(picked_idx, []))

    return [idx_axis_0, idx_axis_1]

