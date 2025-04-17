#!/usr/bin/env python3

from common import get_model, SlidingWindowImage, save_image
import torch
import os
import argparse

parser = argparse.ArgumentParser(prog='train.py')
parser.add_argument('train_dir')
args = parser.parse_args()
train_dir = args.train_dir

if __name__ == "__main__":
    model = get_model()
    device = torch.device("cpu") # torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.load_state_dict(torch.load("model", weights_only=True))

    img_name = sorted([file for file in os.listdir(train_dir) if file.endswith(".png")])[0]
    img_path = os.path.join(train_dir, img_name)
    print(img_path)
    sliding_image = SlidingWindowImage(img_path)
    image_window, target = sliding_image[0]
    print(image_window.shape)
    crop_tensor = image_window.unsqueeze(0).to(device)

    torch.onnx.export(model, crop_tensor, "model.onnx")
    