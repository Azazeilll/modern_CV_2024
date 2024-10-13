from ultralytics import YOLO
import argparse
from pathlib import Path
import torch
import numpy as np
import cv2

def calculate_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / union if union > 0 else 0
    return iou

def load_markup(path, img_shape):
    gt_points = []
    with open(path, 'r') as fin:
        data = fin.readlines()
        for line in data:
            line = line.split(' ')[1:]
            gt_points.append([])
            for i in range(int(len(line)/2)):
                gt_points[-1].append([int(float(line[i*2])*img_shape[1]), 
                                      int(float(line[i*2+1])*img_shape[0])])
    return gt_points

def test_yolo_detections(model_path, config_path, data_dir):
    # Load a model
    model = YOLO(model_path)

    imgs_dir = data_dir / "images"
    markup_dir = data_dir / "labels"
    img_pathes = imgs_dir.glob("*")

    ious = []

    for img_path in img_pathes:
        img = cv2.imread(str(img_path))
        markup_path = markup_dir / (img_path.stem + ".txt")

        markup_pts = load_markup(markup_path, img.shape)
        mask_seg_gt = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for pts in markup_pts:
            pts = np.array(pts)
            mask_seg_gt = cv2.fillPoly(mask_seg_gt, pts=[pts], color=(255))

        results_segmentation = model(img, conf=0.592)

        mask_seg_pred = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for res in results_segmentation:
            if res.masks is None:
                continue
            cur_pts = res.masks[0].xy[0]
            cur_pts = cur_pts.astype(np.int32)
            mask_seg_pred = cv2.fillPoly(mask_seg_pred, pts=[cur_pts], color=(255))

        ious.append(calculate_iou(mask_seg_pred, mask_seg_gt))

    print("Mean IOU:", np.mean(ious))

    metrics = model.val(data=config_path)


def parse_args():
    parser = argparse.ArgumentParser(prog='test')
    parser.add_argument('-m', '--model_path', type=Path)
    parser.add_argument('-c', '--config_path', type=Path)
    parser.add_argument('-d', '--data_dir', type=Path)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    test_yolo_detections(args.model_path, args.config_path, args.data_dir)