import cv2
import os
import torchvision
import torch
import numpy as np
import random
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
                keypoint = tuple(map(int, keypoints[kp, :2].detach().numpy().tolist()))
                # pick the color at the specific color-id
                color = tuple(np.asarray(cmap(color_id[person_id])[:-1])*255)
                # draw a cirle over the keypoint location
                cv2.circle(img_copy, keypoint, 10, color, -1)

    return img_copy



####################################################################################

def get_limbs_from_keypoints(keypoints):
  limbs = [       
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
        [keypoints.index('right_shoulder'), keypoints.index('right_hip')],
        [keypoints.index('left_shoulder'), keypoints.index('left_hip')]
        ]
  return limbs

limbs = get_limbs_from_keypoints(keypoints)
###################################################################################


def draw_skeleton_per_person(img, all_keypoints, all_scores, confs, keypoint_threshold=2, conf_threshold=0.9):
    
    # initialize a set of colors from the rainbow spectrum
    cmap = plt.get_cmap('rainbow')
    # create a copy of the image
    img_copy = img.copy()
    # check if the keypoints are detected
    if len(output["keypoints"])>0:
      # pick a set of N color-ids from the spectrum
      colors = np.arange(1,255, 255//len(all_keypoints)).tolist()[::-1]
      # iterate for every person detected
      for person_id in range(len(all_keypoints)):
          # check the confidence score of the detected person
          if confs[person_id]>conf_threshold:
            # grab the keypoint-locations for the detected person
            keypoints = all_keypoints[person_id, ...]

            # iterate for every limb 
            for limb_id in range(len(limbs)):
              # pick the start-point of the limb
              limb_loc1 = keypoints[limbs[limb_id][0], :2].detach().numpy().astype(np.int32)
              # pick the start-point of the limb
              limb_loc2 = keypoints[limbs[limb_id][1], :2].detach().numpy().astype(np.int32)
              # consider limb-confidence score as the minimum keypoint score among the two keypoint scores
              limb_score = min(all_scores[person_id, limbs[limb_id][0]], all_scores[person_id, limbs[limb_id][1]])
              # check if limb-score is greater than threshold
              if limb_score> keypoint_threshold:
                # pick the color at a specific color-id
                color = tuple(np.asarray(cmap(colors[person_id])[:-1])*255)
                # draw the line for the limb
                cv2.line(img_copy, tuple(limb_loc1), tuple(limb_loc2), color, 10)

    return img_copy


# directory 이름을 분류하고자 하는 클래스 이름으로 할 것. 현재의 index가 출력 index와 동일하게 됨
class_id = ['walk','running','standing','sit','jumping','lie']
DIR = '../action-classifier/video'
WINDOW_SIZE = 9
train_x = ''
train_y = ''
test_x = ''
test_y = ''

# 분류하고자 하는 클래스 각각의 이름
for action in class_id:
    action_dir = os.path.join(DIR,action)
    video_list = os.listdir(action_dir)
    # 행동 폴더 내 비디오 목록
    for video in video_list:
      print(video)
      video_path = os.path.join(action_dir,video)

      # 영상 내 
      cap = cv2.VideoCapture(video_path)
      cnt = 0
      print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      line_x = []
      while cap.isOpened():
        # 프레임 읽어오기
        ret, img = cap.read()
        cnt+=1
        
        print(cnt)
        # 3프레임 중 한 번 마다 통과
        if cnt%3!=0: continue
        if ret:
          # 키포인트 획득
          transform = T.Compose([T.ToTensor()])
          img_tensor = transform(img).cuda()
          output = model([img_tensor])[0]
          '''
          일단 테스트용 영상이 단일 인물이기 때문에 
          단일 인물에 대한 키포인트만 따오지만, 
          추후에 sort 알고리즘을 적용하여 객체별
          추적, 데이터 축적이 필요하다.
          '''
          
          kp=0
          # 아래 문장을 for문으로 대체하면 검출되는 모든 객체에 대한 작업 수행 가능
          # bounding box에 대한 상대적인 키포인트 위치 비율로 변환
          if len(output['scores'])>0:
            if output['scores'][kp]>0.9:
              tmp = output['keypoints'][kp].detach().cpu().numpy().tolist()
              # get flatten
              tmp = [y for x in tmp for y in x[0:2]]
          
              x1 = output['boxes'][kp][0]
              x2 = output['boxes'][kp][2]
              y1 = output['boxes'][kp][1]
              y2 = output['boxes'][kp][3]
                
              box_width = x2-x1
              box_height = y2-y1

              # convert value to ratio of area
              for i in range(len(tmp)):
                if i%2 == 0:
                  tmp[i] -= x1
                  tmp[i] /= box_width
                else:
                  tmp[i] -= y1
                  tmp[i] /= box_height
              line_x.append(','.join("{:.5f}".format(x) for x in tmp))
        else: break
        
    # WINDOW_SIZE에 맞춰서 학습용 데이터 생성
    for i in range(len(line_x)-WINDOW_SIZE):
      _x = line_x[i:i+WINDOW_SIZE]
      if random.random() < 0.7:
        train_x += '\n'.join(_x)
        train_x += '\n'
        train_y += str(class_id.index(action)+1)+'\n'
      else:
        test_x += '\n'.join(_x)
        test_x += '\n'
        test_y += str(class_id.index(action)+1)+'\n'

    cap.release()

f = open("dataset/train_X.txt",'w')
f.write(train_x)
f.close

f = open("dataset/train_Y.txt",'w')
f.write(train_y)
f.close

f = open("dataset/test_X.txt",'w')
f.write(test_x)
f.close

f = open("dataset/test_Y.txt",'w')
f.write(test_y)
f.close
