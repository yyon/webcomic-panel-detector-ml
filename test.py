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

import time

import onnxruntime as ort

from PIL import Image

import argparse

parser = argparse.ArgumentParser(prog='test.py')
parser.add_argument('test_dir')
args = parser.parse_args()
test_dir = args.test_dir

def within_threshold(a, b, threshold):
    return abs(a-b) < threshold

def sliding_window_inference(onnx_session, image_path, threshold=0.5):
    original_image = Image.open(image_path)
    original_width = original_image.width

    sliding_image = SlidingWindowImage(image_path)
    print("testing image", image_path, len(sliding_image), len(sliding_image.samples))

    scale = original_width / sliding_image.target_width

    input_name = onnx_session.get_inputs()[0].name
    output_names = [o.name for o in onnx_session.get_outputs()]  # Expecting ["boxes", "labels", "scores"]

    all_boxes = []

    for i in range(len(sliding_image)):
        image_window, _target = sliding_image[i]
        sample = sliding_image.samples[i]
        top = sample["top"]
        bottom = sample["bottom"]
        print("sample", top, bottom)
        crop_tensor = image_window.unsqueeze(0)

        crop_numpy = crop_tensor.numpy()
        # Run ONNX inference
        outputs = onnx_session.run(output_names, {input_name: crop_numpy})
        boxes, labels, scores = outputs

        print("window:", top, boxes)
        img_name = os.path.basename(sliding_image.image_path)
        test_out_path = os.path.join("test_windows", img_name.replace(".png", "") + "_" + str(i) + ".png")
        save_image(image_window, test_out_path, boxes.tolist())

        # Adjust boxes back to original image coordinates
        for box, score in zip(boxes, scores):
            if score >= threshold:
                x1, y1, x2, y2 = box.tolist()
                y1 = y1 * scale + top
                y2 = y2 * scale + top
                all_boxes.append({
                    "box": [y1, y2],
                    "score": score,
                    "window_i": i,
                    "window": [top, bottom]
                })
    
    to_remove = []

    SAME_LOC_THRESHOLD = original_width * (12/256)
    for box_1 in all_boxes:
        for box_2 in all_boxes:
            if box_1 != box_2 and not (box_1 in to_remove) and not (box_2 in to_remove):
                to_remove_box = None
                if box_1["box"][0] > box_2["box"][0] and box_1["box"][1] < box_2["box"][1]:
                    to_remove_box = box_1
                elif box_2["box"][0] > box_1["box"][0] and box_2["box"][1] < box_1["box"][1]:
                    to_remove_box = box_2
                elif box_1["window_i"] != box_2["window_i"]:
                    same_edges = [within_threshold(box_1["box"][0], box_2["box"][0], SAME_LOC_THRESHOLD), within_threshold(box_1["box"][1], box_2["box"][1], SAME_LOC_THRESHOLD)]
                    box_1_hit_window = [within_threshold(box_1["box"][0], box_1["window"][0], SAME_LOC_THRESHOLD), within_threshold(box_1["box"][1], box_1["window"][1], SAME_LOC_THRESHOLD)]
                    box_2_hit_window = [within_threshold(box_2["box"][0], box_2["window"][0], SAME_LOC_THRESHOLD), within_threshold(box_2["box"][1], box_2["window"][1], SAME_LOC_THRESHOLD)]

                    if same_edges[0]:
                        if box_1_hit_window[1]:
                            to_remove_box = (box_1)
                        elif box_2_hit_window[1]:
                            to_remove_box = (box_2)
                    elif same_edges[1]:
                        if box_1_hit_window[0]:
                            to_remove_box = (box_1)
                        elif box_2_hit_window[0]:
                            to_remove_box = (box_2)

                    if to_remove_box == None and same_edges[0] and same_edges[1]:
                        if box_1["score"] > box_2["score"]:
                            to_remove_box = (box_2)
                        else:
                            to_remove_box = (box_1)

                if to_remove_box != None:
                    to_remove.append(to_remove_box)

    return [[box["box"][0], box["box"][1]] for box in all_boxes if not box in to_remove]

def save_full_image(image_path, path, boxes):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.mkdir(directory)
    shutil.copy(image_path, path)
    if boxes != None:
        with open(path.replace(".png", ".json"), "w") as f:
            f.write(json.dumps([[int(box[0]), int(box[1] - box[0])] for box in boxes]))

if __name__ == "__main__":
    ort_session = ort.InferenceSession("model.onnx")

    for image_name in sorted(os.listdir(test_dir)):
        image_path = os.path.join(test_dir, image_name)

        start = time.time()
        result = sliding_window_inference(ort_session, image_path)
        end = time.time()

        print(image_name, result)
        print("time:", end-start, "s")

        img_name = os.path.basename(image_path)
        test_out_path = os.path.join("test_full", img_name.replace(".png", "") + ".png")
        save_full_image(image_path, test_out_path, result)
