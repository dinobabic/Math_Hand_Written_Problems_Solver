from flask import Flask, render_template, url_for, request, redirect
import base64
import io

import torch
from torch import nn
import requests
import zipfile
from pathlib import Path
import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from timeit import default_timer as timer
import torchvision
import cv2

class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
    super(ResidualBlock, self).__init__()
    self.conv1 = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU())
    self.conv2 = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                    nn.BatchNorm2d(out_channels))
    self.downsample = downsample
    self.relu = nn.ReLU()
    self.out_channels = out_channels

  def forward(self, x):
    residual = x
    out = self.conv1(x)
    out = self.conv2(out)
    if self.downsample:
        residual = self.downsample(x)
    out += residual
    out = self.relu(out)
    return out

class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes = 14):
    super(ResNet, self).__init__()
    self.inplanes = 64
    self.conv1 = nn.Sequential(
                    nn.Conv2d(1, 64, kernel_size = 3, stride = 1, padding = 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU())
    self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
    self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
    self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
    self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
    self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
    self.avgpool = nn.AvgPool2d(2, stride=1)
    self.fc = nn.Linear(4608, num_classes)

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes:

        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
            nn.BatchNorm2d(planes),
        )
    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes
    for i in range(1, blocks):
        layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)


  def forward(self, x):
    x = self.conv1(x)
    x = self.maxpool(x)
    x = self.layer0(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x

def increase_width(cropped_region, new_width):
    # Create a blank white image with the desired width
    new_image = np.ones((cropped_region.shape[0], new_width), dtype=np.uint8) * 255

    # Calculate the position to paste the cropped region
    x_offset = (new_width - cropped_region.shape[1]) // 2

    # Paste the cropped region onto the new image
    new_image[:, x_offset:x_offset + cropped_region.shape[1]] = cropped_region

    return new_image

def split_image_by_black_pixels(image_content, output_list):
    # Convert image content to NumPy array
    image_np = np.frombuffer(image_content, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape

    # Invert the image so that black pixels become white
    inverted_image = cv2.bitwise_not(image)

    # Find contours in the inverted image
    contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by x-coordinate
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

    # Iterate through contours and append regions without black pixels to the list
    cropped_regions = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)

        # Filter out small regions
        if w > 10 and h > 10:  # Adjust the threshold as needed
            # Crop the region without black pixels
            cropped_region = image[y:y+h, x:x+w]
            dict_cropped_region = {'x': x, "y": y, "w":w, "h": h, "is_div": False}

            is_part_of_bigger = False

            for i, tmp_region in enumerate(cropped_regions):
              if (
                  dict_cropped_region["x"] >= tmp_region["x"] - 5
                  and dict_cropped_region["x"] + dict_cropped_region["w"] <= tmp_region["x"] + tmp_region["w"] + 5
              ):
                  tmp_region["y"] = 0
                  tmp_region["h"] = height
                  tmp_region["is_div"] = True

                  cropped_regions[i] = tmp_region
                  is_part_of_bigger = True
                  
            if is_part_of_bigger:
              continue

            cropped_regions.append(dict_cropped_region)

    for region in cropped_regions:
      cropped_region = image[region["y"]:region["y"]+region["h"], region["x"]:region["x"]+region["w"]]
      if (region["is_div"]):
          cropped_region = increase_width(cropped_region, 228)
      output_list.append(cropped_region)
        

def make_predictions(model: torch.nn.Module,
                     image: torch.Tensor,
                     class_names,
                     transform,
                     device):
    model.to(device)
    model.eval()

    with torch.inference_mode():
        image = transform(image.unsqueeze(dim=0).to(device))
        image_logits = model(image)
    image_probs = torch.softmax(image_logits, dim=1)
    image_label = torch.argmax(image_probs, dim=1)

    return class_names[image_label]


model = ResNet(ResidualBlock, [3, 4, 6, 3])
model.load_state_dict(torch.load(f="./simple_photomath_4.pth", map_location=torch.device('cpu')))


transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.Grayscale(num_output_channels=1),
])


class_names = ['div', 'eight', 'five', 'four', 'minus', 'nine', 'one', 'plus', 'seven', 'six', 'three', 'times', 'two', 'zero']



app = Flask(__name__)

def check_if_operation(prediction):
    return prediction == "plus" or prediction == "minus" or prediction == "times" or prediction == "div"
    
def find_number(number):
    if number == "zero":
        return "0"
    if number == "one":
        return "1"
    if number == "two":
        return "2"
    if number == "three":
        return "3"
    if number == "four":
        return "4"
    if number == "five":
        return "5"
    if number == "six":
        return "6"
    if number == "seven":
        return "7"
    if number == "eight":
        return "8"
    if number == "nine":
        return "9"
    return "0"

@app.route("/", methods=["POST", "GET"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            return 'No file part'

        return_image = request.files['file'].read()
        
        output_images = []
        predictions = []
        split_image_by_black_pixels(return_image, output_images)
        
        for image in output_images:
            tensor = torch.from_numpy(image).float() / 255.0
            prediction = make_predictions(model=model,
                                image=tensor.unsqueeze(dim=0),
                                class_names=class_names,
                                transform=transform,
                                device="cpu")
            predictions.append(prediction)
            
        print(predictions)
            
        left_operand = ""
        operation = ""
        right_operand = ""
        found_operation = False
        
        for prediction in predictions:
            if check_if_operation(prediction):
                print(f"Operation is {prediction}")
                found_operation = True
                operation = prediction
                continue
            
            if not found_operation:
                left_operand += find_number(prediction)
            else:
                right_operand += find_number(prediction)
                
        left_operand = int(left_operand)
        right_operand = int(right_operand)
         
        if not found_operation:
            print("Did not find operation")       
        
        result = -1
        if operation == "plus": 
            result = left_operand + right_operand
            operation = "+"
        elif operation == "minus":
            result = left_operand - right_operand
            operation = "-"
        elif operation == "times":
            result = left_operand * right_operand
            operation = "*"
        elif operation == "div":
            result = left_operand / right_operand
            operation = "/"
            
        print(f"{left_operand} {operation} {right_operand} = {result}")

        if return_image:
            encoded_image = base64.b64encode(return_image).decode('utf-8')
            return render_template("index.html", image_content=encoded_image, result=f"{left_operand} {operation} {right_operand} = {result:.2f}" )
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)