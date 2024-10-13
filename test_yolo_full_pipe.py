from ultralytics import YOLO
import argparse
from pathlib import Path
import torch
import cv2
import numpy as np

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

def test_yolo_detections(model_detection_path, model_segmentation_path, data_dir, out_path):
    # Load a model
    model_detection = YOLO(model_detection_path)
    model_segmentation = YOLO(model_segmentation_path)

    imgs_dir = data_dir / "images"
    markup_dir = data_dir / "labels"
    img_pathes = imgs_dir.glob("*")

    ious = []
    imgs = []

    red = (0, 0, 255)

    for img_path in img_pathes:
        img = cv2.imread(str(img_path))
        markup_path = markup_dir / (img_path.stem + ".txt")

        markup_pts = load_markup(markup_path, img.shape)
        mask_seg_gt = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for pts in markup_pts:
            pts = np.array(pts)
            mask_seg_gt = cv2.fillPoly(mask_seg_gt, pts=[pts], color=(255))

        results_detection = model_detection(img, conf=0.308)
        mask_seg_pred = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        img_vis = np.copy(img)

        for result_detection in results_detection:
            xyxy = result_detection.boxes.xyxy.cpu().numpy()
            xyxy = xyxy.astype(np.int32)

            for x1, y1, x2, y2 in xyxy:
                img_vis = cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

                img_cut = img[y1:y2, x1:x2]
                results_segmentation = model_segmentation(img_cut, conf=0.579)
                for res in results_segmentation:
                    # print(res.masks[0])
                    if res.masks is None:
                        continue

                    cur_pts = res.masks[0].xy[0]
                    cur_pts = cur_pts.astype(np.int32)

                    cur_pts[..., 0] += x1
                    cur_pts[..., 1] += y1

                    mask_seg_pred = cv2.fillPoly(mask_seg_pred, pts=[cur_pts], color=(255))

        
        ids = mask_seg_pred > 0
        map_full = np.full(img.shape, (0, 0, 255)) / 2
        img_vis[ids] = 0.5 * img[ids] + map_full[ids]

        for pts in markup_pts:
            pts = np.array(pts)
            img_vis = cv2.polylines(img_vis, [pts], 
                                    True, (255, 0, 0), 2)
        
        iou = calculate_iou(mask_seg_pred, mask_seg_gt)
        ious.append(iou)
        cv2.imwrite(str(out_path / (img_path.stem + f"_{iou}.png")), img_vis)


    print("Mean IOU:", np.mean(ious))


def parse_args():
    parser = argparse.ArgumentParser(prog='test')
    parser.add_argument('-md', '--model_detecton_path', type=Path)
    parser.add_argument('-ms', '--model_segmentation_path', type=Path)
    parser.add_argument('-d', '--data_dir', type=Path)
    parser.add_argument('-o', '--out_dir', type=Path)


    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    test_yolo_detections(args.model_detecton_path, args.model_segmentation_path, args.data_dir, args.out_dir)