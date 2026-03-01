#!/usr/bin/env python3

from common import get_model, SlidingWindowImage, save_image

import os
import json
import torch
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype
from torch.utils.data import Dataset

import torchvision
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
from torchvision.utils import save_image

import albumentations as A
from albumentations.pytorch import ToTensorV2

import argparse

parser = argparse.ArgumentParser(prog='train.py')
parser.add_argument('train_dir')
args = parser.parse_args()
train_dir = args.train_dir

CHECK_TRAINING_DATA = False

if CHECK_TRAINING_DATA:
    if not os.path.exists("training_data_processed"):
        os.mkdir("training_data_processed")

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

        if CHECK_TRAINING_DATA:
            name = f"{img_i}_{sample_i}"
            crop, target = image[sample_i]
            save_image(crop, f"training_data_processed/{name}.png")
            with open(f"training_data_processed/{name}.json", "w") as f:
                regions = []
                for box_i in range(len(target["boxes"])):
                    x1, y1, x2, y2 = target["boxes"][box_i]
                    feature_type = target["labels"][box_i]
                    regions.append([int(y1), int(y2) - int(y1), int(feature_type)])
                f.write(json.dumps(regions))

        return image[sample_i]

def train(model, dataset, device, num_epochs=10):
    model.to(device)
    if os.path.exists("model"):
        print(f"Resuming training from {"model"}")
        model.load_state_dict(torch.load("model", map_location=device))
    else:
        print("Starting training from scratch")

    model.train()
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    for epoch in range(num_epochs):
        data_loader_len = len(data_loader)
        for data_loader_i, data in enumerate(data_loader):
            images, targets = data
            print("train", int((data_loader_i + epoch * data_loader_len) / (data_loader_len * num_epochs) * 100), "(", int(data_loader_i / data_loader_len * 100), ")")
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()

            box_pred = model.roi_heads.box_predictor.bbox_pred
            grad = box_pred.weight.grad  # shape: [4 * num_classes, hidden_dim]
            tx = grad[0::4].abs().mean()
            ty = grad[1::4].abs().mean()
            tw = grad[2::4].abs().mean()
            th = grad[3::4].abs().mean()

            print("(tx, ty, tw, th) = ", tx, ty, tw, th)


            optimizer.step()

        print(f"Epoch {epoch + 1}: Loss = {losses.item():.4f}")
        torch.save(model.state_dict(), "model")

if __name__ == "__main__":
    dataset = SlidingWindowVerticalRegionDataset(train_dir)

    # os.makedirs("train", exist_ok=True)
    # for img_i, sample_i in dataset.images_samples:
    #     full_image = dataset.images[img_i]
    #     img_name = os.path.basename(full_image.image_path)
    #     img, target = full_image[sample_i]
    #     test_out_path = os.path.join("train", img_name.replace(".png", "") + "_" + str(sample_i) + ".png")
    #     save_image(img, test_out_path, target["boxes"])

    device = torch.device("cpu") # torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model()
    train(model, dataset, device)
