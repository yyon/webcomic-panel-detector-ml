#!/usr/bin/env python3

import os
import json
import torch
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype
from torch.utils.data import Dataset

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
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

from PIL import Image

NUM_CLASSES = 2

def get_model():
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT) # torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model

def save_image(image, path, boxes):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.mkdir(directory)
    Image.fromarray(convert_image_dtype(image, dtype=torch.uint8).permute(1, 2, 0).numpy()).save(path)
    if boxes != None:
        with open(path.replace(".png", ".json"), "w") as f:
            f.write(json.dumps([[int(box[1]), int(box[3] - box[1])] for box in boxes]))

class SlidingWindowImage():
    def __init__(self, image_path, window_height=1024, stride=512, target_width=256):
        self.image_path = image_path
        self.target_width = target_width
        self.window_height = window_height
        self.stride = stride

        self.samples = []
        self._prepare_samples()
    
    def _prepare_samples(self):
        img_path = self.image_path
        ann_path = self.image_path.replace(".png", ".json")

        image = read_image(img_path)
        _, h, w = image.shape

        regions = []
        if os.path.exists(ann_path):
            with open(ann_path, "r") as f:
                regions = json.load(f)

        # Convert to [y1, y2]
        regions = [[top, top + height] for top, height in regions]

        # Generate windows
        window_height = round(self.window_height * w / self.target_width)
        stride = round(self.stride * w / self.target_width)
        for top in range(0, h, stride):
            bottom = top + window_height #min(actual_top + window_height, h)
            # top = bottom - window_height
            crop_regions = []

            for y1, y2 in regions:
                # Check if region overlaps with this window
                if y2 > top and y1 < bottom:
                    new_y1 = max(y1, top) - top
                    new_y2 = min(y2, bottom) - top
                    crop_regions.append([0, new_y1, image.shape[2], new_y2])  # full width
            
            if len(crop_regions) > 0 or not os.path.exists(ann_path):
                self.samples.append({
                    "top": top,
                    "bottom": bottom,
                    "regions": crop_regions
                })

            if bottom > h:
                break
    
    def vertical_crop_with_white_fill(self, image: torch.Tensor, top: int, bottom: int) -> torch.Tensor:
        """
        Crop a vertical slice from the image tensor, padding with white if `bottom` exceeds image height.

        Parameters:
        - image: torch.Tensor of shape [3, H, W] with float values in [0, 1]
        - top: Top pixel (inclusive)
        - bottom: Bottom pixel (exclusive), can exceed image height

        Returns:
        - Cropped image tensor of shape [3, bottom - top, W]
        """
        _, H, W = image.shape
        crop_height = bottom - top

        if bottom <= H:
            # Fully in-bounds
            return image[:, top:bottom, :]
        else:
            # Partially or fully out-of-bounds
            in_bounds_crop = image[:, top:H, :]
            pad_height = bottom - H

            # Create white padding [3, pad_height, W] filled with 1.0
            white_pad = torch.ones((3, pad_height, W), dtype=image.dtype, device=image.device)

            return torch.cat([in_bounds_crop, white_pad], dim=1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.image_path
        sample = self.samples[idx]
        image = read_image(img_path)[:3, :, :]  # RGB only
        image = convert_image_dtype(image, dtype=torch.float)

        # Crop vertically
        crop = self.vertical_crop_with_white_fill(image, sample["top"], sample["bottom"])
        print(img_path, sample["top"], sample["bottom"])

        # Albumentations expects HWC numpy arrays
        crop_np = crop.permute(1, 2, 0).numpy()

        # Prepare bounding boxes
        boxes = sample["regions"]  # already in [x1, y1, x2, y2]
        labels = [1] * len(boxes)  # all regions are class 1

        # Apply Albumentations transform
        transform = A.Compose([
            A.Resize(height=self.window_height, width=self.target_width),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        
        # if self.transform:
        transformed = transform(image=crop_np, bboxes=boxes, labels=labels)
        crop = transformed["image"]
        boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
        labels = torch.tensor(transformed["labels"], dtype=torch.int64)
        # else:
        #     # Fallback: convert to tensor if no transform
        #     crop = crop
        #     boxes = torch.tensor(boxes, dtype=torch.float32)
        #     labels = torch.ones((len(boxes),), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }

        return crop, target