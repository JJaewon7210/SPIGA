import cv2
import json
import numpy as np
import torch
from inference.config import ModelConfig
from inference.framework import SPIGAFramework
import copy
from demo.visualize.plotter import Plotter
from colormap import apply_colormap

# Load image and bbox
image = cv2.imread(
    "D:/ThermalData/FaceDB_Snapshot_complete/FaceDB_Snapshot_complete/image_png/irface_sub054_seq07_frm00160.jpg_lfb.png")

x1, y1, x2, y2 = (
    210, 590, 546, 857
)

bbox = (int(x1), int(y1), int(x2-x1), int(y2-y1))

# Process image
dataset = 'merlrav'  # wflw, 300wpublic, 300wprivate, merlrav, cofw68
processor = SPIGAFramework(ModelConfig(dataset))
features = processor.inference(image, [bbox])

# Prepare variables
x0,y0,w,h = bbox
canvas = copy.deepcopy(image)
landmarks = np.array(features['landmarks'][0])
headpose = np.array(features['headpose'][0])

# Plot features
plotter = Plotter()
canvas = plotter.landmarks.draw_landmarks(canvas, landmarks, thick=3)
canvas = plotter.hpose.draw_headpose(canvas, [x0,y0,x0+w,y0+h], headpose[:3], headpose[3:], euler=True)

# Show image results
(h, w) = canvas.shape[:2]
canvas = cv2.resize(canvas, (512, int(h*512/w)))
cv2.imshow('', canvas)
cv2.waitKey()