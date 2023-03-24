import cv2
import json
import numpy as np
import torch
from inference.config import ModelConfig
from inference.framework import SPIGAFramework
import copy
from demo.visualize.plotter import Plotter

# # Load image and bbox
# image = cv2.imread(
#     # "D:/ThermalData/Charlotte_ThermalFace/S1/N12187.jpg")
# "D:/ThermalData/FaceDB_Snapshot_complete/FaceDB_Snapshot_complete/image_png/irface_sub001_seq02_frm00155.jpg_lfb.png")
# # image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
# bbox = (127, 142, 223 - 127, 252 - 142)
# bbox = (193, 447, 480-193, 738-447)


# # Process image
# dataset = 'merlrav'  # wflw, 300wpublic, 300wprivate, merlrav, cofw68
# processor = SPIGAFramework(ModelConfig(dataset))
# features = processor.inference(image, [bbox])



# # Prepare variables
# x0,y0,w,h = bbox
# canvas = copy.deepcopy(image)
# landmarks = np.array(features['landmarks'][0])
# headpose = np.array(features['headpose'][0])

# # Plot features
# plotter = Plotter()
# canvas = plotter.landmarks.draw_landmarks(canvas, landmarks)
# canvas = plotter.hpose.draw_headpose(canvas, [x0,y0,x0+w,y0+h], headpose[:3], headpose[3:], euler=True)

# # Show image results
# (h, w) = canvas.shape[:2]
# canvas = cv2.resize(canvas, (512, int(h*512/w)))
# cv2.imshow('', canvas)
# cv2.waitKey(10000)
# # cv2.imwrite("D:/Users/Jaewon/Pictures/irface_sub001_seq02_frm00155_black.jpg_lfb.jpg", canvas)

# # # 
# Load image and bbox
image = cv2.imread(
    # "D:/ThermalData/Charlotte_ThermalFace/S1/N12187.jpg")
    "D:/ThermalData/FaceDB_Snapshot_complete/FaceDB_Snapshot_complete/image_png/irface_sub001_seq02_frm00155.jpg_lfb.png")
# image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
bbox = (127, 142, 223 - 127, 252 - 142)
bbox = (193, 447, 480-193, 738-447)

# Process image
dataset = 'merlrav'  # wflw, 300wpublic, 300wprivate, merlrav, cofw68
processor = SPIGAFramework(ModelConfig(dataset))

image_resize = cv2.resize(image, (256, 256))
image_tensor = torch.from_numpy(image_resize).unsqueeze(0)
bboxes = np.array([bbox])
bboxes[:, 0] = bboxes[:, 0] * 256/image.shape[1]
bboxes[:, 1] = bboxes[:, 1] * 256/image.shape[0]
bboxes[:, 2] = bboxes[:, 2] * 256/image.shape[1]
bboxes[:, 3] = bboxes[:, 3] * 256/image.shape[0]
bbox_tensor = torch.from_numpy(bboxes)

features, outputs = processor.pred(image_tensor, bbox_tensor)

# Prepare variables
x0, y0, w, h = bbox
canvas = copy.deepcopy(image)
landmarks = outputs['Landmarks'][-1]
landmarks = landmarks[0].cpu().detach().numpy()
landmarks[:, 0] = landmarks[:, 0]*image.shape[1]
landmarks[:, 1] = landmarks[:, 1]*image.shape[0]

headpose = outputs['Poses'][-1]
headpose = headpose[0].cpu().detach().numpy()

# Plot features
plotter = Plotter()
canvas = plotter.landmarks.draw_landmarks(canvas, landmarks)
canvas = plotter.hpose.draw_headpose(
    canvas, [x0, y0, x0+w, y0+h], headpose[:3], headpose[3:], euler=True)

# Show image results
(h, w) = canvas.shape[:2]
canvas = cv2.resize(canvas, (512, int(h*512/w)))
cv2.imshow('', canvas)
cv2.waitKey(10000)
