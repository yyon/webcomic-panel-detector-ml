#!/usr/bin/env python3

import os
import shutil
import json
import torch
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype
from torch.utils.data import Dataset

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from torchvision.ops import nms

import os
import json
import torch
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from common import get_model, SlidingWindowImage, save_image

import argparse

parser = argparse.ArgumentParser(prog='test.py')
parser.add_argument('test_dir')
args = parser.parse_args()
test_dir = args.test_dir

def sliding_window_inference(model, image_path, window_height=1024, stride=512, threshold=0.5):
    model.eval()
    device = next(model.parameters()).device

    sliding_image = SlidingWindowImage(image_path)
    print("testing image", image_path, len(sliding_image), len(sliding_image.samples))

    all_boxes = []
    all_scores = []

    for i in range(len(sliding_image)):
        image_window, _target = sliding_image[i]
        sample = sliding_image.samples[i]
        top = sample["top"]
        bottom = sample["bottom"]
        print("sample", top, bottom)
        crop_tensor = image_window.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(crop_tensor)[0]
            print("window:", top, outputs["boxes"])
            img_name = os.path.basename(sliding_image.image_path)
            test_out_path = os.path.join("test_windows", img_name.replace(".png", "") + "_" + str(i) + ".png")
            save_image(image_window, test_out_path, outputs["boxes"])

        # Adjust boxes back to original image coordinates
        for box, score in zip(outputs['boxes'], outputs['scores']):
            if score >= threshold:
                x1, y1, x2, y2 = box.tolist()
                all_boxes.append([x1, y1 + top, x2, y2 + top])
                all_scores.append(score)

    if all_boxes:
        boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(all_scores)
        keep = nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
        final_boxes = boxes_tensor[keep].tolist()
        return final_boxes
    else:
        return []

def save_full_image(image_path, path, boxes):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.mkdir(directory)
    shutil.copy(image_path, path)
    if boxes != None:
        with open(path.replace(".png", ".json"), "w") as f:
            f.write(json.dumps([[int(box[1]), int(box[3] - box[1])] for box in boxes]))


if __name__ == "__main__":
    model = get_model()
    device = torch.device("cpu") # torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.load_state_dict(torch.load("model", weights_only=True))
    for image_name in sorted(os.listdir(test_dir)):
        image_path = os.path.join(test_dir, image_name)

        result = sliding_window_inference(model, image_path)

        print(image_name, result)

        img_name = os.path.basename(image_path)
        test_out_path = os.path.join("test_full", img_name.replace(".png", "") + ".png")
        save_full_image(image_path, test_out_path, result)
