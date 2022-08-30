# Pose Estimation using LSTM

brighten-internship 폴더에 action-classifier, train-lstm, train-keypoints 세 가지 디렉토리.

* train-keypoints

  학습용 데이터의 키포인트 검출 정확도가 낮을 경우 fine tunning하기 위해서 작성한 키포인트 학습 프로젝트

* train-lstm

  long-short term memory 기법을 사용하여 시계열 데이터로 움직이는 keypoint의 시간에 따른 정보 배열을 두고 어떤 행동인지 분류 학습하는 프로젝트

* action-classifier

  학습 결과를 확인할 수 있도록 만든 gui 프로그램



### train-lstm

영상에서 keypoint를 검출하고 학습용 데이터로 쌓은 후 lstm 분류 학습을 진행하여 가중치 파일을 얻어오는 일련의 과정을 수행하기 위한 프로젝트

사용한 프레임워크는 pytorch의 keypoint-rcnn과 lstm이며, `get_data_from_video.py  ` 의 영상이 저장된 디렉토리인 `DIR`, 분류할 대상을 나열한 리스트 `class_id`, 시계열 데이터의 단위를 지정하는 `WINDOW_SIZE`를 수정하여 원하는 방식으로 커스텀 할 수 있다.

변하는 이미지 해상도에 공통적으로 적용하기 위해서 keypoint의 위치를 절대 좌표가 아닌 box 영역에 대한 상대적인 비율로 나타내도록 변경했다.



**입력 예시**

```python
# 구분하고자 하는 동작을 모두 나열
class_id = ['walk', 'running', 'standing', 'sit', 'jumping', 'lie']
# 영상이 위치한 경로 입력 / 위의 class_id와 하위 디렉토리가 같아야 함
DIR = '../action-classifier/video'
# 분류에 사용할 시계열 데이터 누적 프레임 수
WINDOW_SIZE = 9
```

**실행 결과**

```
X_train.txt
0.37912,0.12413,···,0.89685
0.36207,0.10943,···,0.83670
0.40556,0.12842,···,0.92979
0.46635,0.11873,···,0.91137
0.51230,0.10714,···,0.93027
0.47177,0.10870,···,0.93144
0.40756,0.11056,···,0.92574
0.41089,0.11489,···,0.92718
0.44690,0.11022,···,0.90895
⋮
```

```
Y_train.txt
1
⋮
```



X, Y에 대한 train과 test dataset을 모두 구축했으면, 

https://colab.research.google.com/drive/1yg6SQ8iWNRS8yz1Gum9znIAZmd_C_3hy#scrollTo=Uendd5wdvYAA

해당 링크의 코드를 실행시켜서 

가중치 파일 model.pt를 얻어올 수 있다.

메모에 어떻게 실행할 수 있는지 작성해 놓았다.
