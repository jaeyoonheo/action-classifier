# action-classifier

사람이 존재하는 영상으로부터 joint를 검출하여 특징점의 프레임별 시계열 데이터로 lstm을 학습, 어떤 행동을 하고있는 것인지 판별하는 프로그램

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

### 앞으로 해야하는 것
현재 Tracking 알고리즘으로 부여된 객체 ID와 얼굴 이미지를 매칭해서 저장. 현재까지 수행한 것들의 DB 전송용 Dictionary 데이터 생성
검출된 사람 정보에서 얼굴 검출
+ 기존에 Face Recognize를 섞어서 Id구분을 하려고 했으나 cctv에서는 정면도 아니고 얼굴의 크기도 작아서 특징점 추출이 어려울 것으로 간주됨. 따라서 얼굴을 통한 id 구분은 보류
+ Face Detection을 수행하려면 따로 모델을 load 해야하고 그만큼 자원 소모가 늘어나는데, 그냥 keypoint에서 nose, eyes, ears 기준으로 하면 편할듯.
+ 이라고 생각했는데 하다보니 SORT 이전에 keypoint 기반 얼굴을 따야하는데 그게 조금 힘들다.
