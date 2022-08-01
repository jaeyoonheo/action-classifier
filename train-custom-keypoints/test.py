import cv2
import os
import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt

import json

from torch.utils.data import Dataset, DataLoader

from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F

import albumentations as A # Library for augmentations

import transforms, utils, engine, train
from utils import collate_fn
from engine import train_one_epoch, evaluate

import time


################################################################################
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_model(num_keypoints, weights_path=None):
    
    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False,
                                                                   pretrained_backbone=True,
                                                                   num_keypoints=num_keypoints,
                                                                   num_classes = 2, # Background is the first class, object is the second class
                                                                   rpn_anchor_generator=anchor_generator)

    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)        
        
    return model

model = get_model(num_keypoints = 27)
model.to(device)

#params = [p for p in model.parameters() if p.requires_grad]
#optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

model.load_state_dict(torch.load("keypointsrcnn_weights.pth"))



# create a model object from the keypointrcnn_resnet50_fpn class
#model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
# call the eval() method to prepare the model for inference mode.
model.eval()
#model.cuda()

# create the list of keypoints.
keypoints = ['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow',
                'right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee', 'right_knee', 'left_ankle','right_ankle']


###############################################################################


# import the transforms module
from torchvision import transforms as T


####################################################################################

def draw_keypoints_per_person(img, all_keypoints, all_scores, confs, keypoint_threshold=2, conf_threshold=0.9):
    # initialize a set of colors from the rainbow spectrum
    cmap = plt.get_cmap('rainbow')
    # create a copy of the image
    img_copy = img.copy()
    # pick a set of N color-ids from the spectrum
    color_id = np.arange(1,255, 255//len(all_keypoints)).tolist()[::-1]
    # iterate for every person detected
    for person_id in range(len(all_keypoints)):
      # check the confidence score of the detected person
      if confs[person_id]>conf_threshold:
        # grab the keypoint-locations for the detected person
        keypoints = all_keypoints[person_id, ...]
        # grab the keypoint-scores for the keypoints
        scores = all_scores[person_id, ...]
        # iterate for every keypoint-score
        for kp in range(len(scores)):
            # check the confidence score of detected keypoint
            if scores[kp]>keypoint_threshold:
                # convert the keypoint float-array to a python-list of intergers
                keypoint = tuple(map(int, keypoints[kp, :2].detach().cpu().numpy().tolist()))
                # pick the color at the specific color-id
                color = tuple(np.asarray(cmap(color_id[person_id])[:-1])*255)
                # draw a cirle over the keypoint location
                cv2.circle(img_copy, keypoint, 3, color, -1)

    return img_copy



####################################################################################


# video_path = '2-4_001-C01.mp4'

# video = cv2.VideoCapture(video_path)    

# index = 0
# while video.isOpened():
#   ret, img = video.read()

#   if img is None:
#     break
  
#   transform = T.Compose([T.ToTensor()])
#   img_tensor = transform(img).cuda()
  
#   output = model([img_tensor])[0]

#   #limbs = get_limbs_from_keypoints(keypoints)
  
#   output_keypoint = 'output/keypoint-' + str(index).zfill(5) + '.jpg'
#   #output_skeleton = 'output/skeleton-' + str(index).zfill(5)
  
#   keypoints_img = draw_keypoints_per_person(img, output["keypoints"], output["keypoints_scores"], output["scores"],keypoint_threshold=2)
#   cv2.imwrite(output_keypoint, keypoints_img)
#   #skeletal_img = draw_skeleton_per_person(img, output["keypoints"], output["keypoints_scores"], output["scores"],keypoint_threshold=2)
#   #cv2.imwrite(output_skeleton, skeletal_img)
#   index += 1

#   print(index)
  
img_list = os.listdir("input")


index = 0
for img_path in img_list:
  start = time.time()
  img = cv2.imread('input/'+ img_path,cv2.IMREAD_COLOR)
  if img is None:
    break
  
  transform = T.Compose([T.ToTensor()])
  img_tensor = transform(img).cuda()
  
  output = model([img_tensor])[0]
  print(1/(time.time()-start))
  #limbs = get_limbs_from_keypoints(keypoints)
  
  output_keypoint = 'output/keypoint-' + str(index).zfill(5) + '.jpg'
  #output_skeleton = 'output/skeleton-' + str(index).zfill(5)
  
  keypoints_img = draw_keypoints_per_person(img, output["keypoints"], output["keypoints_scores"], output["scores"],keypoint_threshold=2)
  cv2.imwrite(output_keypoint, keypoints_img)
  #skeletal_img = draw_skeleton_per_person(img, output["keypoints"], output["keypoints_scores"], output["scores"],keypoint_threshold=2)
  #cv2.imwrite(output_skeleton, skeletal_img)
  index += 1

  print(index)
  