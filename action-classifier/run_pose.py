import cv2
import os
import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt


################################################################################

# create a model object from the keypointrcnn_resnet50_fpn class
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
# call the eval() method to prepare the model for inference mode.
model.eval()
model.cuda()


# create the list of keypoints.
keypoints = ['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow',
                'right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee', 'right_knee', 'left_ankle','right_ankle']

# import the transforms module
from torchvision import transforms as T

video_list = os.listdir('./video')
lstm_model = torch.load("model.pt")
lstm_model.eval()
# results = ''
results = [0,0,0,0,0,0]

for video_path in video_list:
    
    # results += video_path +'\n'
    index = 0
    pose_queue=[]
    # Read the image using opencv 
    img_path = './video/'+video_path
    video = cv2.VideoCapture(img_path)
    print(video)
    cnt = 0
    while video.isOpened():
        cnt+=1
        if cnt%3!=0:
            continue
        ret, img = video.read()
        
        if (img is None):
            break
        transform = T.Compose([T.ToTensor()])
        img_tensor = transform(img).cuda()
        
        output = model([img_tensor])[0]
        
        for kp in range(len(output["keypoints"])):
            if output["scores"][kp]>0.001:
                tmp = output["keypoints"][kp].detach().cpu().numpy().tolist()
                tmp = [y for x in tmp for y in x]

                tmp_pos = []
                for i in range(len(tmp)):
                    if i%3 != 2:
                        tmp_pos.append(tmp[i])
                pose_queue.append(tmp_pos)
                
                if len(pose_queue) > 9:
                    pose_queue.pop(0)
                    temp=[]
                    temp.append(pose_queue)
                    temp = torch.Tensor(temp)
                    print(temp)
                    result = lstm_model(temp)
                    result = result.tolist()[0]
                    result = result.index(max(result))
                    results[result] += 1
                    # results = results + str(result+1) +'\n'

print(results)

# f = open('results.txt','w')
# f.write(results)
# f.close()
    