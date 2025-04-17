#!/usr/bin/env python3

import os
import json
import torch
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype
from torch.utils.data import Dataset

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
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

parser = argparse.ArgumentParser(prog='train.py')
parser.add_argument('train_dir')
args = parser.parse_args()
train_dir = args.train_dir

class SlidingWindowVerticalRegionDataset(Dataset):
    def __init__(self, directory, window_height=1024, stride=512, target_width=256):
        self.directory = directory
        self.target_width = target_width
        self.window_height = window_height
        self.stride = stride

        self.images = []
        self.images_samples = []
        image_files = sorted([file for file in os.listdir(self.directory) if file.endswith(".png")])
        for img_i, img_name in enumerate(image_files):
            img_path = os.path.join(self.directory, img_name)
            image = SlidingWindowImage(img_path, window_height, stride, target_width)
            self.images.append(image)
            for sample_i in range(len(image)):
                self.images_samples.append([img_i, sample_i])

    def __len__(self):
        return len(self.images_samples)

    def __getitem__(self, idx):
        img_i, sample_i = self.images_samples[idx]
        image = self.images[img_i]
        return image[sample_i]

def train(model, dataset, device, num_epochs=10):
    model.to(device)
    model.train()
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    for epoch in range(num_epochs):
        for images, targets in data_loader:
            print("train", [img.shape for img in images], targets)
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}: Loss = {losses.item():.4f}")
        torch.save(model.state_dict(), "model")

def sliding_window_inference(model, image, window_height=1024, stride=512, threshold=0.5):
    model.eval()
    device = next(model.parameters()).device
    _, h, w = image.shape

    all_boxes = []
    all_scores = []

    for top in range(0, h, stride):
        bottom = min(top + window_height, h)
        crop = image[:, top:bottom, :]
        crop_tensor = crop.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(crop_tensor)[0]

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

if __name__ == "__main__":
    dataset = SlidingWindowVerticalRegionDataset(train_dir)

    for img_i, sample_i in dataset.images_samples:
        full_image = dataset.images[img_i]
        img_name = os.path.basename(full_image.image_path)
        img, target = full_image[sample_i]
        test_out_path = os.path.join("train", img_name.replace(".png", "") + "_" + str(sample_i) + ".png")
        save_image(img, test_out_path, target["boxes"])

    device = torch.device("cpu") # torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model()
    train(model, dataset, device)
