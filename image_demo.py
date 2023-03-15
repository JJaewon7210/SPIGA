import cv2
import json
import numpy as np

# Load image and bbox
image = cv2.imread(
    "D:/ThermalData/Integrate/Crop/image_jpg/irface_sub001_seq02_frm00460.jpg_lfb.jpg")
# image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
bbox = (0,0, image.shape[1], image.shape[0])

from inference.config import ModelConfig
from inference.framework import SPIGAFramework

# Process image
dataset = 'merlrav'  # wflw, 300wpublic, 300wprivate, merlrav, cofw68
processor = SPIGAFramework(ModelConfig(dataset))
features = processor.inference(image, [bbox])

import copy
from demo.visualize.plotter import Plotter

# Prepare variables
x0,y0,w,h = bbox
canvas = copy.deepcopy(image)
landmarks = np.array(features['landmarks'][0])
headpose = np.array(features['headpose'][0])

# Plot features
plotter = Plotter()
canvas = plotter.landmarks.draw_landmarks(canvas, landmarks)
canvas = plotter.hpose.draw_headpose(canvas, [x0,y0,x0+w,y0+h], headpose[:3], headpose[3:], euler=True)

# Show image results
(h, w) = canvas.shape[:2]
canvas = cv2.resize(canvas, (512, int(h*512/w)))
cv2.imshow('', canvas)
cv2.waitKey(10000)
# cv2.imwrite("D:/Users/Jaewon/Pictures/irface_sub001_seq02_frm00155_black.jpg_lfb.jpg", canvas)