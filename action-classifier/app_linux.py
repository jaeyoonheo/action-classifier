# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'videoplayer.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!

from socket import SOCK_SEQPACKET
import cv2, os, threading
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QListWidget, QListWidgetItem, QSlider
from PyQt5.QtGui import QPixmap
import numpy as np
from math import dist

import cv2
import csv
import datetime
import os
import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms as T

import joint
import distance

from sort import *


# action class / 학습할 때 사용한 index와 동일해야 함
# action = ['walk','running','standing','sit','jumping','lie']
action = ['lie','running','sit','walk']

# create a model object from the keypointrcnn_resnet50_fpn class
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
# call the eval() method to prepare the model for inference mode.
model.eval()
model.cuda()

# create model instance for classify action
lstm_model = torch.load("./model/2022-09-16.pt")
lstm_model.eval()

# 동영상 쓰레드에서 재생 상태를 확인하기 위한 global 변수
running = False
changed = False
currentFrame = 0


class Ui_MainWindow(object):
    '''
    UI 구성
    '''
    def setupUi(self, MainWindow):
        self.th = threading.Thread(target=self.run, daemon=True)
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1400, 1000)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(10, 10, 1380, 950))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.preImg = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.preImg.setObjectName("preImg")
        self.horizontalLayout_2.addWidget(self.preImg)
        self.processedImg = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.processedImg.setObjectName("processedImg")
        self.horizontalLayout_2.addWidget(self.processedImg)
        self.verticalLayout_4.addLayout(self.horizontalLayout_2)
        self.videoSlider = QtWidgets.QSlider(self.verticalLayoutWidget_2)
        self.videoSlider.setOrientation(QtCore.Qt.Horizontal)
        self.videoSlider.setObjectName("videoSlider")
        self.videoSlider.valueChanged.connect(self.moved_slider)
        self.verticalLayout_4.addWidget(self.videoSlider)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.prevBtn = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.prevBtn.setObjectName("prevBtn")
        self.prevBtn.clicked.connect(self.prev)
        self.horizontalLayout_3.addWidget(self.prevBtn)
        self.playBtn = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.playBtn.setObjectName("playBtn")
        self.playBtn.clicked.connect(self.start)
        self.horizontalLayout_3.addWidget(self.playBtn)
        self.nextBtn = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.nextBtn.setObjectName("nextBtn")
        self.nextBtn.clicked.connect(self.next)
        self.horizontalLayout_3.addWidget(self.nextBtn)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.videoList = QtWidgets.QListWidget(self.verticalLayoutWidget_2)
        self.videoList.setObjectName("videoList")
        self.videoList.itemDoubleClicked.connect(self.itemClicked)
        self.verticalLayout.addWidget(self.videoList)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.newBtn = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.newBtn.setObjectName("newBtn")
        self.newBtn.clicked.connect(self.new)
        self.horizontalLayout_4.addWidget(self.newBtn)
        self.delBtn = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.delBtn.setObjectName("delBtn")
        self.delBtn.clicked.connect(self.delete)
        self.horizontalLayout_4.addWidget(self.delBtn)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.verticalLayout_4.addLayout(self.verticalLayout)
        self.frameNum = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.frameNum.setObjectName("frameNum")
        self.verticalLayout_4.addWidget(self.frameNum)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 687, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # Value init
        self.jointQueue = []

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.preImg.setText(_translate("MainWindow", "TextLabel"))
        self.processedImg.setText(_translate("MainWindow", "TextLabel"))
        self.prevBtn.setText(_translate("MainWindow", "<"))
        self.playBtn.setText(_translate("MainWindow", "Play"))
        self.nextBtn.setText(_translate("MainWindow", ">"))
        self.newBtn.setText(_translate("MainWindow", "New"))
        self.delBtn.setText(_translate("MainWindow", "Delete"))
        self.frameNum.setText(_translate("MainWindow", "TextLabel"))
        
    ''' 
    영상 재생
    '''
    def run(self):
        global running, changed, currentFrame

        # 멀티 트래킹을 위한 객체 생성
        mot_tracker = Sort()

        # 현재 재생 중인 영상의 카운트를 저장하기 위한 변수
        cnt = 0
        # 영상 load
        self.cap = cv2.VideoCapture(self.video_path)
        # 영상 정보 get
        frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 영상 재생 현황을 나타내기 위한 slider의 현재 위치 세팅
        self.videoSlider.setRange(0,frame-1)
        self.videoSlider.setSingleStep(1)
        personal_report = []
        # 영상이 열려 있으면
        while self.cap.isOpened():
            cnt += 1
            ret, img = self.cap.read()
            # 확인해야 하는 프레임 단위 지정. 현재 영상은 초당 30프레임, 인식할 영상은 초당 10프레임으로 3번당 한 번씩 인식하도록 구성
            if cnt%3 != 0:
                continue
            # 영상이 정상적으로 read 되었을 때
            if ret:
                self.videoSlider.setValue(self.cap.get(cv2.CAP_PROP_POS_FRAMES)-1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h,w,c = img.shape
                # 영상을 label에 출력하기 위해 크기 설정
                qImg = QtGui.QImage(img.data, w, h, w*c, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qImg)
                        
                # 이미지를 텐서로 변환
                transform = T.Compose([T.ToTensor()])
                img_tensor = transform(img).cuda()
                        
                # 모델에 영상을 입력해서 pose detection 결과를 확인
                output = model([img_tensor])[0]
                skeletal_img = joint.draw_skeleton_per_person(img, output["keypoints"], output["keypoints_scores"], output["scores"],keypoint_threshold=2)    
                print('현재 프레임 : '+str(cnt))

                # SORT Object Tracking
                detections = []
                kp_list = []
                for kp in range(len(output["scores"])):
                    if output["scores"][kp]>0.9:
                        detections.append(output["boxes"][kp].detach().cpu().numpy().tolist())
                        detections[-1].append(output["scores"][kp].detach().cpu().numpy().tolist())
                        kp_list.append(kp)
                track_bbs_ids = mot_tracker.update(detections)
                
                # 객체 추적 결과는 id의 역순으로 저장되므로 뒤에서부터 실행
                for i in reversed(range(len(track_bbs_ids))):
                    id = int(track_bbs_ids[i][4])
                    x = int(track_bbs_ids[i][0])
                    y = int(track_bbs_ids[i][1])

                    # 검출된 키포인트 결과를 box의 크기에 대한 비율로 변환
                    kp = kp_list[i]
                    tmp = output["keypoints"][kp].detach().cpu().numpy().tolist()
                    tmp = [y for x in tmp for y in x[0:2]]
                    
                    '''
                    tmp[0,1] 코, tmp[2,3] 왼쪽눈, tmp[4,5] 오른쪽 눈, tmp[6,7] 왼쪽 귀, tmp[8,9] 오른쪽 
                    코에서 전체 영상의 1/6 size만큼 얼굴 crop
                    시계열 데이터가 기준치 이상이면 거리 측정 실행.
                    '''
                    boxes = output['boxes'][kp].cpu() # .numpy().tolist()

                    x1 = boxes[0]
                    x2 = boxes[2]
                    y1 = boxes[1]
                    y2 = boxes[3]

                    box_width = x2-x1
                    box_height = y2-y1

                    face_std = max(box_width, box_height) //12
                    face_std = int(face_std)

                    face_x1 = int(tmp[0] - face_std)
                    face_y1 = int(tmp[1] - face_std)
                    face_x2 = int(tmp[0] + face_std)
                    face_y2 = int(tmp[1] + face_std)

                    for i in range(len(tmp)):
                        if i%2 == 0:
                            tmp[i] -= x1
                            tmp[i] /= box_width
                        else:
                            tmp[i] -= y1
                            tmp[i] /= box_height

                    # 총 9개의 프레임까지 저장하고, 그 갯수가 넘어가면 lstm 모델에 입력해서 어떤 동작인지 결과 확인
                    # 분류에 사용하는 시계열 데이터의 길이를 늘렸을 때 정확도가 비약적으로 향상되는 것을 확인함
                    # result는 학습된 클래스 개수만큼 각 동작에 대한 confidence를 출력 / 가장 높은 confidence를 가지는 값을 영상에 출력
                    if len(self.jointQueue) < id :
                        self.jointQueue.append([tmp])
                        start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        # id, 입장 시간, 퇴장 시간, 얼굴 경로
                        personal_report.append([id, start, start, 0,'not detected'])

                        prev_dis, prev_theta = distance.distance_angle_measure((x1+x2)/2, y2)

                    else :
                        self.jointQueue[id-1].append(tmp)

                        # 퇴장 시간 갱신
                        personal_report[id-1][2] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # 카메라 정보와 프레임 정보는 distance.py에서 수정
                        # 이동 거리 갱신
                        prev_dis, prev_theta, traveled_distance = distance.get_traveled(prev_dis, prev_theta, (x1+x2)/2, y2)
                        # 이동 거리 합산
                        personal_report[id-1][3] += traveled_distance

                        if len(self.jointQueue[id-1])>8:
                            result = lstm_model(torch.Tensor([self.jointQueue[id-1]])).tolist()
                            result = result[0]
                            print(result)
                            result = result.index(max(result))
                            self.jointQueue[id-1].pop(0)
                            
                            cv2.putText(skeletal_img, str(id)+' '+action[result], (x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(177,100,192),5,cv2.LINE_AA)

                            # 얼굴 없으면
                            if personal_report[id-1][4] == 'not detected':
                                # 얼굴 나오는 지 검사
                                if not (tmp[0] > tmp[6] and tmp[0] < tmp[8]):
                                    # 얼굴 이미지 크롭 후 경로 저장
                                    face_img = img[face_y1:face_y2,face_x1:face_x2].copy()
                                    face_dir = './save/id-'+str(id)+'.jpg'
                                    cv2.imwrite(face_dir,face_img)
                                    personal_report[id-1][4] = face_dir
                            continue
                    cv2.putText(skeletal_img, str(id), (x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(177,100,192),5,cv2.LINE_AA)
                
                # joint가 검출된 영상을 라벨에 출력
                sqImg = QtGui.QImage(skeletal_img.data,w,h,w*c,QtGui.QImage.Format_RGB888)
                spixmap = QtGui.QPixmap.fromImage(sqImg)
                        
                # 출력 영상 resize
                p = pixmap.scaled(int(w*480/h), 480, QtCore.Qt.IgnoreAspectRatio)
                sp = spixmap.scaled(int(w*480/h), 480, QtCore.Qt.IgnoreAspectRatio)
                self.preImg.setPixmap(p)
                self.processedImg.setPixmap(sp)

                f = open('personal_log.csv','w', newline = '')
                writer = csv.writer(f)
                writer.writerows(personal_report)
                f.close()
                
            else: break
        # 영상 재생 끝난 후 다시 돌아보기
        # 이전 코드 일부 반복
        changed = True
        preFrame = currentFrame
        while changed:
            if preFrame != currentFrame:
                preFrame = currentFrame
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, currentFrame-1)
                print(currentFrame)
                ret,img = self.cap.read()
                if ret:
                    h,w,c = img.shape
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    qImg = QtGui.QImage(img.data, w, h, w*c, QtGui.QImage.Format_RGB888)
                    pixmap = QtGui.QPixmap.fromImage(qImg)
                    p = pixmap.scaled(int(w*480/h), 480, QtCore.Qt.IgnoreAspectRatio)
                    img_tensor = transform(img).cuda()
                        
                    output = model([img_tensor])[0]
                    skeletal_img = joint.draw_skeleton_per_person(img, output["keypoints"], output["keypoints_scores"], output["scores"],keypoint_threshold=2)    
                    sqImg = QtGui.QImage(skeletal_img.data,w,h,w*c,QtGui.QImage.Format_RGB888)
                    spixmap = QtGui.QPixmap.fromImage(sqImg)

                    sp = spixmap.scaled(int(w*480/h), 480, QtCore.Qt.IgnoreAspectRatio)
                    self.preImg.setPixmap(p)
                    self.processedImg.setPixmap(sp)

        self.cap.release()
        self.jointQueue=[]
        # 인스턴스 할당 해제
        del mot_tracker
        # create instance of SORT

        print("Thread end.")

    # play 버튼 클릭 시 이벤트
    def start(self):
        global running
        # 최초 실행 시 flag 체크해서 중복재생 방지
        if not self.th.is_alive():
            running = True
            # thread 생성 후 시작
            self.th = threading.Thread(target=self.run, daemon=True)
            self.th.start()
            print("started")
            return
        # 이미 thread가 실행되어 있다면 중복 실행을 방지하기 위해 아무 행동도 하지 않음
        print("now exist thread")

    def onExit(self):
        print("exit")
        self.stop()
        
    # 원래 이전 프레임 확인 용도로 작성했으나 기능x
    def prev(self):
        global running
        running = False
        
    # 원래 이후 프레임 확인 용도로 작성했으나 기능x
    def next(self):
        global running
        running = True
    
    # new 버튼을 클릭했을 때 경로를 입력하는 함수
    def new(self):
        filename, extension = QtWidgets.QFileDialog.getOpenFileName(None,'Open File')
        self.videoList.addItem(QListWidgetItem(filename))

    # new로 추가한 경로를 리스트에서 삭제하는 함수
    def delete(self):
        lst_modelindex = self.videoList.selectedIndexes()

        for modelindex in lst_modelindex:
            print(modelindex.row())
            self.videoList.model().removeRow(modelindex.row())
    
    # 리스트의 아이템이 더블 클릭 되었을 때 실행 / 쓰레드 종료 글로벌 변수 변경
    def itemClicked(self):
        global running, changed
        running = False
        changed = False
        self.video_path = self.videoList.selectedItems()[0].text()
        # self.cap = cv2.VideoCapture(self.video_path)
        print(self.video_path)

    # 슬라이더가 움직일 때 해당 위치로 영상 이동
    def moved_slider(self, value):
        global running, changed, currentFrame
        currentFrame = value
        changed = True


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
