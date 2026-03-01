import argparse

import pytesseract
import cv2

from PIL import Image
import numpy as np

from enum import Enum
import os
import json

parser = argparse.ArgumentParser(prog='editor.py')
parser.add_argument('images_dir')
args = parser.parse_args()
images_dir = args.images_dir

class FeatureType(int, Enum):
    PANEL = 0
    TEXTBOX = 1
    SPEECH_UP = 2
    SPEECH_DOWN = 3

def floodfill_box(image_np, seed_x, seed_y):

    fill_color = 155

    flags = (
        4 |                      # connectivity
        cv2.FLOODFILL_MASK_ONLY |
        cv2.FLOODFILL_FIXED_RANGE |
        (255 << 8)
    )

    res = cv2.floodFill(
        image_np,
        None,
        (seed_x, seed_y),
        fill_color,
        loDiff=40,
        upDiff=40,
        flags=flags
    )

    (num_pixels, image, mask, rect) = res

    return rect

def x_regions(image_np, regions):
    # print("img size", image_np.shape)
    
    original_width = image_np.shape[1]
    original_height = image_np.shape[0]

    img_width = 1024
    img_height = int(original_height * img_width / original_width)
    img = cv2.resize(image_np[:, :, :3], dsize=(img_width, img_height), interpolation=cv2.INTER_CUBIC)

    nearby_tolerance = int(img_width) * 0.05

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bw = (gray > 40).astype(np.uint8) * 255

    new_regions = []
    for original_region in regions:
        unscaled_y1 = original_region["y1"]
        unscaled_y2 = original_region["y2"]
        feature_type = original_region["featureType"]
        if feature_type == FeatureType.PANEL:
            x1 = 0
            x2 = original_width
        else:
            scaled_y1 = unscaled_y1 * img_height / original_height
            scaled_y2 = unscaled_y2 * img_height / original_height

            img_cut = gray[int(scaled_y1):int(scaled_y2), :]

            # print("img_cut", img_cut, int(scaled_y1), int(scaled_y2))
            results = pytesseract.image_to_data(img_cut, output_type=pytesseract.Output.DICT)

            # print("results", results)
            if not results:
                return

            boxes = []

            for i in range(len(results["conf"])):
                conf = results["conf"][i]
                x1 = int(results["left"][i])
                x2 = x1 + int(results["width"][i])
                y1 = int(results["top"][i])
                y2 = y1 + int(results["height"][i])
                text = results["text"][i]
                # print(conf > 50 and len(text.strip()) > 0, text, conf, x1, x2, y1, y2)
                if conf > 50 and len(text.strip()) > 0:
                    boxes.append([x1, y1 + scaled_y1, x2, y2 + scaled_y2])
            # print("boxes", boxes)

            # Floodfill + add as TEXTBOX regions
            best_fit = None
            for x1, y1, x2, y2 in boxes:
                mid_y = int((y1 + y2) / 2)
                mid_x = int((x1 + x2) / 2)

                seed_y = max(int(y1 - img_width * 0.01), 0)
                seed_x = mid_x

                rx, ry, rw, rh = floodfill_box(gray, seed_x, seed_y)

                difference = abs(ry - scaled_y1) + abs(ry + rh - scaled_y2)

                if best_fit == None or difference < best_fit[0]:
                    best_fit = [difference, ry, ry + rh, rx, rx + rw]

            if best_fit == None or best_fit[0] > img_width * 0.2:
                print("backup strategy")
                best_fit = None
                if len(boxes) == 0:
                    for i in range(50):
                        seed_x = int(img_width * i/50)
                        seed_y = int((scaled_y1 + scaled_y2)/2)

                        rx, ry, rw, rh = floodfill_box(gray, seed_x, seed_y)

                        difference = abs(ry - scaled_y1) + abs(ry + rh - scaled_y2)

                        if best_fit == None or difference < best_fit[0]:
                            best_fit = [difference, ry, ry + rh, rx, rx + rw]

            if best_fit == None or best_fit[0] > img_width * 0.2:
                print("backup strategy 2")
                best_fit = None
                if len(boxes) == 0:
                    for i in range(50):
                        seed_x = int(img_width * i/50)
                        seed_y = int((scaled_y1 + scaled_y2)/2)

                        rx, ry, rw, rh = floodfill_box(bw, seed_x, seed_y)

                        difference = abs(ry - scaled_y1) + abs(ry + rh - scaled_y2)

                        if best_fit == None or difference < best_fit[0]:
                            best_fit = [difference, ry, ry + rh, rx, rx + rw]
                        
            if best_fit == None or best_fit[0] > img_width * 0.2:
                x1 = 0
                x2 = original_width
            else:
                x1 = best_fit[3] * original_width / img_width
                x2 = best_fit[4] * original_width / img_width
        new_regions.append([int(unscaled_y1), int(unscaled_y2) - int(unscaled_y1), feature_type, int(x1), int(x2) - int(x1)])
    
    return new_regions


image_files = [
    os.path.join(images_dir, f)
    for f in os.listdir(images_dir)
    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
]
image_files.sort()

for image_file in image_files:
    print(image_file)
    json_file = image_file.rsplit(".", 1)[0] + ".json"
    if not os.path.exists(json_file):
        continue
    with open(json_file, "r") as f:
        json_text = f.read()
    
    regions = json.loads(json_text)
    print("original", regions)
    regions = [{"y1": region[0], "y2": region[1] + region[0], "featureType": 0 if len(region) == 2 else region[2]} for region in regions]
    print(regions)
    img = Image.open(image_file) # Load image
    img_np = np.asarray(img)

    new_regions = x_regions(img_np, regions)
    print(image_file)
    print("x regions", new_regions)

    with open(json_file, "w") as f:
        json_text = f.write(json.dumps(new_regions))
