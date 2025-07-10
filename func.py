import random
import cv2
import numpy as np

MODEL_INPUT_SIZE = 1024
OBJ_THRESH, NMS_THRESH = 0.25, 0.45
CLASS_NAMES = ("rbd", "ejg", "rb", "panel")
class_colors = [[random.randint(0, 255) for _ in range(3)] for _ in CLASS_NAMES]

def decode_output(output_data):
    data = output_data[0]
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    strides = [8, 16, 32]
    grid_sizes = [MODEL_INPUT_SIZE // s for s in strides]
    num_preds = [g * g for g in grid_sizes]
    total_preds = sum(num_preds)
    
    predictions = data.reshape(9, -1).T
    
    all_boxes, all_classes, all_scores = [], [], []
    start_idx = 0
    
    for stride, grid_size, num_pred in zip(strides, grid_sizes, num_preds):
        scale_data = predictions[start_idx:start_idx+num_pred]
        start_idx += num_pred
        
        grid_y, grid_x = np.mgrid[0:grid_size, 0:grid_size]
        grid_x = grid_x.flatten() * stride
        grid_y = grid_y.flatten() * stride
        
        tx, ty = scale_data[:, 0], scale_data[:, 1]
        cx = (sigmoid(tx) * 2 - 0.5 + grid_x)
        cy = (sigmoid(ty) * 2 - 0.5 + grid_y)
        
        tw, th = scale_data[:, 2], scale_data[:, 3]
        width = (sigmoid(tw) * 2) ** 2 * stride * 4
        height = (sigmoid(th) * 2) ** 2 * stride * 4
        
        obj_score = sigmoid(scale_data[:, 4])
        cls_scores = sigmoid(scale_data[:, 5:5+len(CLASS_NAMES)])
        class_ids = np.argmax(cls_scores, axis=1)
        max_cls_score = np.max(cls_scores, axis=1)
        confidence = obj_score * max_cls_score
        
        x1 = np.clip(cx - width/2, 0, MODEL_INPUT_SIZE)
        y1 = np.clip(cy - height/2, 0, MODEL_INPUT_SIZE)
        x2 = np.clip(cx + width/2, 0, MODEL_INPUT_SIZE)
        y2 = np.clip(cy + height/2, 0, MODEL_INPUT_SIZE)
        
        all_boxes.append(np.column_stack([x1, y1, x2, y2]))
        all_classes.append(class_ids)
        all_scores.append(confidence)
    
    boxes = np.vstack(all_boxes)
    classes = np.concatenate(all_classes)
    scores = np.concatenate(all_scores)
    
    return boxes, classes, scores

def nms_boxes(boxes, scores):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    padding = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    
    coords[:, [0, 2]] -= padding[0]
    coords[:, [1, 3]] -= padding[1]
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)

def clip_coords(boxes, img_shape):
    boxes[:, 0] = np.clip(boxes[:, 0], 0, img_shape[1])
    boxes[:, 1] = np.clip(boxes[:, 1], 0, img_shape[0])
    boxes[:, 2] = np.clip(boxes[:, 2], 0, img_shape[1])
    boxes[:, 3] = np.clip(boxes[:, 3], 0, img_shape[0])

def yolov8_post_process(output_data):
    boxes, classes, scores = decode_output(output_data)
    
    mask = scores > OBJ_THRESH
    boxes = boxes[mask]
    classes = classes[mask]
    scores = scores[mask]
    
    if len(boxes) > 0:
        keep = nms_boxes(boxes, scores)
        boxes = boxes[keep]
        classes = classes[keep]
        scores = scores[keep]
    
    return boxes, classes, scores

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def draw_results(img, boxinfo):
    for xyxy, conf, cls in boxinfo:
        label = f'{CLASS_NAMES[int(cls)]} {conf:.2f}'
        plot_one_box(xyxy, img, label=label, color=class_colors[int(cls)], line_thickness=1)
    return img

def letterbox(im, new_shape=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), color=(0, 0, 0)):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (left, top)

def object_detection_function(rknn_model, input_image):
    processed_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    processed_image, ratio, padding = letterbox(processed_image, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    processed_image = np.expand_dims(processed_image, 0)
    
    outputs = rknn_model.inference(inputs=[processed_image], data_format=['nhwc'])
    
    boxes, classes, scores = yolov8_post_process(outputs)
    
    if boxes is not None and len(boxes) > 0:
        scale_coords((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), boxes, input_image.shape[:2])
        draw_results(input_image, zip(boxes, scores, classes))
    return input_image
