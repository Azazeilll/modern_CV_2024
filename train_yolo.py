from ultralytics import YOLO
import argparse
from pathlib import Path
import torch

def train_yolo_detections(config_path: Path, base_yolo_model: str):
    # Load a model
    model = YOLO(base_yolo_model)  # load a pretrained model (recommended for training)

    # Train the model with 2 GPUs
    results = model.train(data=config_path, epochs=300, imgsz=640, device=[0, 1])


def parse_args():
    parser = argparse.ArgumentParser(prog='train')
    parser.add_argument('-c', '--config_path', type=Path)
    parser.add_argument('-b', '--base_yolo_model', type=str)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    train_yolo_detections(args.config_path, args.base_yolo_model)
