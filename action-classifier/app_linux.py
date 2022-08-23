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
import os
import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms as T

import joint

from sort import *


# action class
action = ['walk', 'run', 'hug', 'collapse', 'cross arms', 'clap']

# create a model object from the keypointrcnn_resnet50_fpn class
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
# call the eval() method to prepare the model for inference mode.
model.eval()
model.cuda()

lstm_model = torch.load("./model/lstm_model.pt")
lstm_model.eval()

# create the list of keypoints.

def kp_flat(input):
    input_flatten = [y for x in input for y in x]
    tmp_pos = []
    for i in range(len(input_flatten)):
        if i%3 != 2 :
            tmp_pos.append(input_flatten[i])
    return tmp_pos

running = False
changed = False
currentFrame = 0

class Ui_MainWindow(object):
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

        # create instance of SORT
        mot_tracker = Sort()
        
        cnt = 0
        self.cap = cv2.VideoCapture(self.video_path)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.videoSlider.setRange(0,frame-1)
        self.videoSlider.setSingleStep(1)
        strg = []

        while self.cap.isOpened():
            cnt += 1
            ret, img = self.cap.read()
            if ret:
                strg.append([])
                self.videoSlider.setValue(self.cap.get(cv2.CAP_PROP_POS_FRAMES)-1)
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                sleep_ms = int(np.round((1/fps)*500))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h,w,c = img.shape
                qImg = QtGui.QImage(img.data, w, h, w*c, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qImg)
                        
                transform = T.Compose([T.ToTensor()])
                img_tensor = transform(img).cuda()
                        
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
                        print(detections)
                        kp_list.append(kp)
                track_bbs_ids = mot_tracker.update(detections)
                print(track_bbs_ids)
                
                for i in reversed(range(len(track_bbs_ids))):
                    id = int(track_bbs_ids[i][4])
                    x = int(track_bbs_ids[i][0])
                    y = int(track_bbs_ids[i][1])
                    tmp = kp_flat(output["keypoints"][kp_list[i]].detach().cpu().numpy().tolist())
                    if len(self.jointQueue) < id :
                        self.jointQueue.append([tmp])
                    else :
                        self.jointQueue[id-1].append(tmp)
                        if len(self.jointQueue[id-1])>8:
                            result = lstm_model(torch.Tensor([self.jointQueue[id-1]])).tolist()
                            result = result.index(max(result))
                            self.jointQueue[id-1].pop(0)
                            print(result)
                            cv2.putText(skeletal_img, str(id)+'   '+action[result], (x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(177,100,192),5,cv2.LINE_AA)
                            continue
                    cv2.putText(skeletal_img, str(id), (x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(177,100,192),5,cv2.LINE_AA)


                sqImg = QtGui.QImage(skeletal_img.data,w,h,w*c,QtGui.QImage.Format_RGB888)
                spixmap = QtGui.QPixmap.fromImage(sqImg)
                        
                # 출력 영상 resize
                p = pixmap.scaled(int(w*480/h), 480, QtCore.Qt.IgnoreAspectRatio)
                sp = spixmap.scaled(int(w*480/h), 480, QtCore.Qt.IgnoreAspectRatio)
                self.preImg.setPixmap(p)
                self.processedImg.setPixmap(sp)

                
            else: break
        # 영상 재생 끝난 후 다시 돌아보기
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
        print("Thread end.")
    
    def stop(self):
        print("stoped")
        
    def start(self):
        global running
        # 최초 실행 시 flag 체크해서 중복재생 방지
        if not self.th.is_alive():
            running = True
            self.th = threading.Thread(target=self.run, daemon=True)
            self.th.start()
            print("started")
            return
        print("now exist thread")

    def onExit(self):
        print("exit")
        self.stop()
        
    def prev(self):
        global running
        running = False
        
    def next(self):
        global running
        running = True
        
    def new(self):
        filename, extension = QtWidgets.QFileDialog.getOpenFileName(None,'Open File')
        self.videoList.addItem(QListWidgetItem(filename))

    def delete(self):
        lst_modelindex = self.videoList.selectedIndexes()

        for modelindex in lst_modelindex:
            print(modelindex.row())
            self.videoList.model().removeRow(modelindex.row())
            
    def itemClicked(self):
        global running, changed
        running = False
        changed = False
        self.video_path = self.videoList.selectedItems()[0].text()
        # self.cap = cv2.VideoCapture(self.video_path)
        print(self.video_path)

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
