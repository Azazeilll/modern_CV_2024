import argparse
from pathlib import Path
import cv2
import numpy as np

EPS = 15

def parse_args():
    parser = argparse.ArgumentParser(prog='test')
    parser.add_argument('-d', '--data_dir', type=Path)

    return parser.parse_args()

def load_markup(markup_path, img_shape):
    with open(markup_path, "r") as f:
        markup = f.readlines()

    markup_new = []
    for obj_markup in markup:

        obj_markup = obj_markup.split(" ")

        if obj_markup[-1] == "\n":
            obj_markup = obj_markup[:-1]

        cls = obj_markup[0]

        pixels_map = np.array(obj_markup[1:], dtype=np.float32)
        pixels_map = pixels_map.reshape(int(len(pixels_map) / 2), 2)

        pixels_map[..., 0] *= (img_shape[1] - 1)
        pixels_map[..., 1] *= (img_shape[0] - 1)

        markup_new.append({"cls": cls, "pixels_map": pixels_map})

    return markup_new

def save_markup(markup, save_path, img_shape):
    with open(save_path, "w") as f:
        for obj_markup in markup:

            pixels_map = obj_markup["pixels_map"]
            pixels_map = pixels_map.astype(np.float32)

            pixels_map[..., 0] /= (img_shape[1] - 1)
            pixels_map[..., 1] /= (img_shape[0] - 1)

            pixels_map = pixels_map.flatten()
            pixels_map_str = " ".join(pixels_map.astype(str))
            f.write(str(obj_markup["cls"]) + " " + pixels_map_str + "\n")

def vis_markup(img, markup, out_path=None):
    img_vis = img.copy()
    color = (0, 0, 255)
    thickness = 3
    isClosed = True

    for obj_markup in markup:
        pts = obj_markup["pixels_map"].astype(np.int32)
        pts = pts.reshape((-1, 1, 2))
        img_vis = cv2.polylines(img_vis, [pts], isClosed, color, thickness)

    if out_path is not None:
        cv2.imwrite(out_path, img_vis)
    else:
        return img_vis

def cut_bboxes(input_dir):
    imgs_dir = input_dir / "images"
    imgs_out_dir = input_dir / "images_cut"
    if not imgs_out_dir.exists():
        imgs_out_dir.mkdir()

    markup_dir = input_dir / "labels"
    markup_out_dir = input_dir / "labels_cut"
    if not markup_out_dir.exists():
        markup_out_dir.mkdir()

    pathes_list = imgs_dir.glob("*")


    for img_path in pathes_list:
        img = cv2.imread(str(img_path))

        markup_path = markup_dir / (img_path.stem + '.txt')
        markup = load_markup(markup_path, img.shape)
        # vis_markup(img, markup, "./tmp.png")

        for i, obj in enumerate(markup):
            pts = obj["pixels_map"].astype(np.int32)
            x0, y0 = pts[..., 0].min(), pts[..., 1].min()
            x1, y1 = pts[..., 0].max(), pts[..., 1].max()

            y0 = max(0, y0 - EPS)
            y1 = min(img.shape[0], y1 + EPS)

            x0 = max(0, x0 - EPS)
            x1 = min(img.shape[1], x1 + EPS)

            pts[..., 0] -= x0
            pts[..., 1] -= y0
            markup_new = [{"pixels_map" : pts, "cls" : obj["cls"]}]

            img_cut = img[y0:y1, x0:x1]
            save_markup(markup_new, markup_out_dir / f"{img_path.stem}_{i}.txt", img_cut.shape)

            vis_markup(img_cut, markup_new, "./tmp.png")
            cv2.imwrite(str(imgs_out_dir / f"{img_path.stem}_{i}.png"), img_cut)


if __name__ == "__main__":
    args = parse_args()

    dirs_list = list(args.data_dir.iterdir())

    for d in dirs_list:
        if not d.is_dir() and not d == args.data_dir:
            continue

        cut_bboxes(d)