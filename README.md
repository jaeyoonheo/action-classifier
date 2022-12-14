# action-classifier

사람이 존재하는 영상으로부터 joint를 검출하여 특징점의 프레임별 시계열 데이터로 lstm을 학습, 어떤 행동을 하고있는 것인지 판별하는 프로그램
사용을 위해서 학습된 LSTM model.pt와 얼굴 유사도 검사 siamese model.pt가 필요합니다
LSTM의 경우 학습을 위해서 Human-Pose-Estimation를 이용합니다. https://github.com/spmallick/learnopencv

Siamese 얼굴 유사도 검사를 위해서 이 repository를 참고합니다. https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch


### Keypoint Detection
pytorch의 keypoint-rcnn pretrained model을 사용하여 인물으로부터 17개의 신체부위 keypoint를 검출한다.
이미지 혹은 영상을 통해서 읽어올 수 있고, output으로는 각 keypoint의 x,y 좌표, confidence, bounding box의 좌상, 우하 좌표, confidence를 얻을 수 있다.
검출된 keypoint 중에서 높은 confidence를 갖는 것들을 시계열 데이터로 저장한다.

### Action Classification by LSTM
마찬가지로 pytorch에서 제공하는 LSTM 구조를 사용한다.
Keypoint detection을 통해 얻은 keypoint 좌표 시계열 데이터를 일정 길이 단위로 학습시켜 어떤 행동인지 파악한다.

### Custom Keypoint Training
기존의 17점 keypoint을 대신하여 사용하기 위해 16점 혹은 27점 keypoint를 직접 학습한다.
annotation file과 image file이 1:1로 대칭되어야 하며, 폴더 내부 README.md에 학습 양식을 적어놓았다.

### Latest Update

얼굴 이미지를 crop해서 서버에 저장하고 
(id, 입장 시간, 퇴장 시간, 이동 거리, 얼굴 이미지 경로) 를 csv 파일로 저장하도록 업데이트 했습니다.

모든 검출된 객체들을 DB에 저장할 수 있도록 파일을 출력합니다. save 폴더에서 크롭된 얼굴 이미지를 확인할 수 있습니다.



> [다크 프로그래머 :: 영상의 기하학적 해석 - 영상의 지면 투영(ground projection) (tistory.com)](https://darkpgmr.tistory.com/153

해당 포스팅을 참고해서 영상에서 사람이 닿아있는 지면까지의 거리와 각도, cosine 법칙의 대변의 길이를 구하는 공식을 통해 이전 프레임의 위치에서 현재 프레임의 위치까지 거리를 갱신하고 합산하도록 적용했습니다.

Face Detection의 경우 Pytorch에서 FaceNet 모델을 적용했으나, 기존 Keypoint RCNN의 결과물로 얼굴의 keypoint를 지정하고 있기에 자원을 절약하기 위해 정면 / 측면을 바라보는 얼굴을 crop할 수 있도록 수정했습니다.

SORT 알고리즘을 통해 Object Tracking을 수행하고 있으며 개별 객체마다 id를 부여하고 입장, 퇴장시간, 이동 거리, 얼굴 이미지를 저장, 갱신합니다.



### 수정 가능한 변수

카메라의 calibration 정보와 frame size를 갱신해야 이동 거리를 정상적으로 산출할 수 있습니다.

`action_classifier/app_linux.py` 의 LSTM_size 변수를 수정할 수 있습니다.

입력하는 시계열 데이터의 길이가 길어지면 정확도가 향상되지만 반응 속도가 감소합니다.

32Frame을 사용했을 때 약 89%의 정확도를 가지고, 동작 1초 후에 반응합니다.



`action_classifier/app_linux.py` 의 Line 280에서 이미지의 저장 경로를 수정할 수 있습니다.



`action_classifier/distance.py`  에서 수정할 수 있습니다.

![image](https://user-images.githubusercontent.com/109254266/194532029-edd00cda-f87f-4cb0-b639-291f77246f4b.png)
얼굴 검출 확인


# Compare to Face ID in Database 

### Added Function

* `action_classifier.py`, `app_linux.py`  : DB에 저장된 얼굴 이미지와 현재 프레임에서 검출된 객체의 얼굴 이미지를 비교, 대상의 id와 이동 경로, 입장 및 퇴장 시간을 DB에 기록하는 기능 추가

* code line 312 ~ 364



### Reference Stack

* Siamese : 얼굴에서 특징점을 검출하고 두 이미지 사이의 유사도를 검사 (낮을수록 닮은 사람)
* csv : csv 파일 출력



### Changed

* SiameseNetwork 구조 클래스 추가



### Usage

* requirements.txt를 install 한 후 동작
* 기능을 사용하기 위해 아래 링크를 통해 모델을 새로 학습할 수 있음
  https://colab.research.google.com/github/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch/blob/master/Siamese-networks-medium.ipynb 
* 얼굴 이미지를 저장하기 위해 save 폴더 추가 필요
* faceDB에 기존에 저장된 인원들의 얼굴 이미지를 추가



### Output

![image](https://user-images.githubusercontent.com/109254266/198023594-4d24f727-a8e1-4d8e-af58-8a33b5dc4171.png)

좌측에서부터 각 id, entrance time, exit time, moved distance, directory of face image, matched id, similarity



![image](https://user-images.githubusercontent.com/109254266/198023876-54e7cc64-d6c1-4874-ae67-58b9634c837d.png)

