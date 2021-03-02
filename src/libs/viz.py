import cv2
import numpy as np

from src.libs.utils import apply_nms


def integrate_frame(events, leak, frame_h, frame_w, prev_output):
    y, x, ts = events.T
    if prev_output is None:
        frame = np.zeros([frame_h, frame_w], np.float32)
        prev_ts = 0
    else:
        frame, prev_ts = prev_output

    new_frame = frame.copy()

    # Updates the frame
    last_event_ts = np.max(ts)
    new_frame -= (last_event_ts - prev_ts) * leak
    new_frame[new_frame < 0] = 0
    new_frame[y, x] += 1 - (last_event_ts - ts) * leak
    new_frame[new_frame < 0] = 0

    return new_frame, last_event_ts


def convert_bboxes(bboxes, grid_h, grid_w, h_image, w_image, sqrt):
    cell_idx_h = np.arange(grid_h, dtype=np.float32)
    cell_idx_w = np.arange(grid_w, dtype=np.float32)
    # Creates two [grid_h, grid_w] matrices containing in each position the corresponding
    # row (and column) index.
    # Eg: (2x2): row: [[0, 0], [1, 1]],  col: [[0, 1], [0, 1]]
    col_idx = np.reshape(np.tile(cell_idx_w, [grid_h]), [grid_h, grid_w])  # [grid_h, grid_w]
    row_idx = np.tile(np.expand_dims(cell_idx_h, axis=-1), [1, grid_w])  # [grid_h, grid_w]

    # Reshapes the indices so that numpy can correctly broadcast the operations.
    col_idx = np.reshape(col_idx, [1, grid_h, grid_w, *([1] * (bboxes.ndim - 3))])
    row_idx = np.reshape(row_idx, [1, grid_h, grid_w, *([1] * (bboxes.ndim - 3))])

    true_x = ((bboxes[..., 0:1] + col_idx) / grid_w) * w_image  # [num_frames, S, S, B, 1]
    true_y = ((bboxes[..., 1:2] + row_idx) / grid_h) * h_image  # [num_frames, S, S, B, 1]
    true_w = (np.square(bboxes[..., 2:3]) if sqrt else bboxes[..., 2:3]) * w_image  # [num_frames, S, S, B, 1]
    true_h = (np.square(bboxes[..., 3:4]) if sqrt else bboxes[..., 3:4]) * h_image  # [num_frames, S, S, B, 1]
    transformed_bboxes = np.concatenate([true_x, true_y, true_w, true_h], axis=-1)  # [num_frames, S, S, B, 4]

    return transformed_bboxes


def draw_bboxes_cv2(images, batch_bboxes, batch_valid, batch_labels, batch_confs, max_thickness=5, resize_ratio=1,
                    color=np.array([1., 1., 1.]), transparency=True, highlight_top_n=0,
                    highlight_color=np.array([0., 0., 1.]), label_bottom=False):

    img_width, img_height = images.shape[1:3]

    drawn_images = []
    # Loops over the batch dimension
    for image, bboxes, bvalid, blabel, bconf in zip(images, batch_bboxes, batch_valid, batch_labels, batch_confs):
        valid_bbox = bboxes[bvalid]
        valid_label = blabel[bvalid]
        valid_conf = bconf[bvalid]

        # Sort by confidence
        sorted_idx = np.argsort(valid_conf)[::-1]
        sorted_bbox = valid_bbox[sorted_idx]
        sorted_label = valid_label[sorted_idx]
        sorted_conf = valid_conf[sorted_idx]

        img_tmp = image.copy()
        # Normalizes the image in [0, 1] range
        cv2.normalize(img_tmp, img_tmp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # Extends the image to RGB
        if img_tmp.shape[-1] == 1:
            img_tmp = np.repeat(img_tmp, 3, axis=-1)

        # Loops over the boxes to be drawn
        for n, (bbox, label, conf) in enumerate(zip(sorted_bbox, sorted_label, sorted_conf)):

            # Computes the top-left and bottom-right points
            x_center, y_center, w, h = bbox
            x_min = int(x_center - w / 2)
            y_min = int(y_center - h / 2)
            x_max = int(x_center + w / 2)
            y_max = int(y_center + h / 2)

            # We need to use an additional image if we want the transparency effect because
            # OpenCV does not allow to specify alpha values
            black_img = np.zeros_like(img_tmp, dtype=np.float32)

            label = label.decode("utf-8") if isinstance(label, bytes) else label
            conf = np.clip(conf, 0.0, 1.0)
            thickness = np.floor(conf * max_thickness).astype(np.int32)
            # Maps values from (0.0, 1.0) to (0.2, 1.0)
            alpha = (conf - 0.0) * (1.0 - 0.2) / (1.0 - 0.0) + 0.2
            alpha = alpha if transparency else 1.0

            # If the bounding box is one of the top n, use the provided color
            col = highlight_color if n < highlight_top_n else color
            txt = "{} {}%".format(label, int(conf * 100)) if n < highlight_top_n else label

            # Draws the bounding box (1px border)
            cv2.rectangle(black_img, (x_min - 1, y_min - 1), (x_max + 1, y_max + 1), col, thickness)

            # Draws the label
            (w, h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            h += 1
            y_txt_bottom = y_min - 2 if not label_bottom else y_max + h + 2
            cv2.rectangle(black_img, (x_min - 1, y_txt_bottom - h), (x_min + w - 1, y_txt_bottom), col, -1)
            cv2.putText(black_img, txt, (x_min - 1, y_txt_bottom - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0))

            # Merges the two images based on the alpha value
            img_tmp = cv2.addWeighted(img_tmp, 1.0, black_img, alpha, 0, dtype=cv2.CV_32F)

        if resize_ratio != 1:
            img_tmp = cv2.resize(img_tmp, (img_height * resize_ratio, img_width * resize_ratio),
                                 interpolation=cv2.INTER_NEAREST)

        drawn_images.append(img_tmp)

    drawn_images = np.stack(drawn_images, axis=0).astype(np.float32)

    return drawn_images


def draw_bboxes(net_predictions, frame, h_grid, w_grid, num_classes, idx_to_label=None,
                conf_threshold=0.2, use_nms=False, nms_threshold=0.2, max_thickness=5, highlight_top_n=0,
                resize_ratio=1):

    h_image, w_image = frame.shape

    net_predictions = np.expand_dims(net_predictions, axis=0)
    frames = frame.reshape([1, h_image, w_image, 1])
    num_frames = 1

    pred_label = net_predictions[..., :num_classes]  # [num_frames, S, S, C]
    pred_bbox = net_predictions[..., num_classes:]
    pred_bbox = np.reshape(pred_bbox, [num_frames, h_grid, w_grid, -1, 5])  # [num_frames, S, S, B, 5]

    pred_bbox_params = pred_bbox[..., 0:4]  # [num_frames, S, S, B, 4]
    pred_bbox_conf = pred_bbox[..., 4:5]  # [num_frames, S, S, B, 1]
    pred_trans_bbox = convert_bboxes(pred_bbox_params, h_grid, w_grid, h_image, w_image, sqrt=True)

    # Multiplies the box's confidence by the cell's label
    pred_label = np.expand_dims(pred_label, axis=-2) * pred_bbox_conf  # [batch_size, S, S, B, C]

    # Flats the grid's structure
    pred_trans_bbox = np.reshape(pred_trans_bbox, [num_frames, -1, 4])  # [num_frames, num_bbox, 4]
    pred_label = np.reshape(pred_label, [num_frames, -1, num_classes])  # [num_frames, num_bbox, num_classes]
    pred_bbox_conf = np.reshape(np.max(pred_bbox_conf, axis=-1), [num_frames, -1])  # [num_frames, num_bbox]
    pred_valid_bbox = pred_bbox_conf > conf_threshold  # [num_frames, num_bbox]

    if use_nms and np.any(pred_valid_bbox):
        picked_idx = apply_nms(pred_trans_bbox, pred_bbox_conf, pred_valid_bbox, iou_threshold=nms_threshold)
        mask = np.zeros_like(pred_bbox_conf, dtype=np.bool)
        if picked_idx[0].size > 0:
            mask[picked_idx] = True
        padding = np.zeros_like(pred_bbox_conf, np.float32)
        pred_bbox_x1 = np.where(mask, pred_trans_bbox[..., 0], padding)
        pred_bbox_y1 = np.where(mask, pred_trans_bbox[..., 1], padding)
        pred_bbox_x2 = np.where(mask, pred_trans_bbox[..., 2], padding)
        pred_bbox_y2 = np.where(mask, pred_trans_bbox[..., 3], padding)
        pred_trans_bbox = np.stack([pred_bbox_x1, pred_bbox_y1, pred_bbox_x2, pred_bbox_y2], axis=-1)
        pred_bbox_conf = np.where(mask, pred_bbox_conf, padding)
        pred_valid_bbox = np.where(mask, pred_valid_bbox, padding.astype(np.bool))

    if idx_to_label is None:
        idx_to_label = np.arange(num_classes).astype(np.str)
    pred_idx = np.argmax(pred_label, axis=-1)
    pred_label = idx_to_label[pred_idx]

    # ### Draw frames ### #
    drawn_frames = draw_bboxes_cv2(frames, pred_trans_bbox, pred_valid_bbox, pred_label, pred_bbox_conf,
                                   max_thickness=max_thickness, resize_ratio=resize_ratio,
                                   highlight_top_n=highlight_top_n, highlight_color=np.array([0., 0., 1.]))

    return drawn_frames
