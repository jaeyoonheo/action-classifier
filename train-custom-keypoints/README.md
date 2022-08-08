# Object detection reference training scripts

This folder contains reference training scripts for object detection.
They serve as a log of how to train specific models, to provide baseline
training and evaluation scripts to quickly bootstrap research.

To execute the example commands below you must install the following:

```
cython
pycocotools
matplotlib
```

You must modify the following flags:

`--data-path=/path/to/coco/dataset`

`--nproc_per_node=<number_of_gpus_available>`

Except otherwise noted, all models have been trained on 8x V100 GPUs. 

### Faster R-CNN ResNet-50 FPN
```
torchrun --nproc_per_node=8 train.py\
    --dataset coco --model fasterrcnn_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3 --weights-backbone ResNet50_Weights.IMAGENET1K_V1
```

### Faster R-CNN MobileNetV3-Large FPN
```
torchrun --nproc_per_node=8 train.py\
    --dataset coco --model fasterrcnn_mobilenet_v3_large_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3 --weights-backbone MobileNet_V3_Large_Weights.IMAGENET1K_V1
```

### Faster R-CNN MobileNetV3-Large 320 FPN
```
torchrun --nproc_per_node=8 train.py\
    --dataset coco --model fasterrcnn_mobilenet_v3_large_320_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3 --weights-backbone MobileNet_V3_Large_Weights.IMAGENET1K_V1
```

### FCOS ResNet-50 FPN
```
torchrun --nproc_per_node=8 train.py\
    --dataset coco --model fcos_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3  --lr 0.01 --amp --weights-backbone ResNet50_Weights.IMAGENET1K_V1
```

### RetinaNet
```
torchrun --nproc_per_node=8 train.py\
    --dataset coco --model retinanet_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3 --lr 0.01 --weights-backbone ResNet50_Weights.IMAGENET1K_V1
```

### SSD300 VGG16
```
torchrun --nproc_per_node=8 train.py\
    --dataset coco --model ssd300_vgg16 --epochs 120\
    --lr-steps 80 110 --aspect-ratio-group-factor 3 --lr 0.002 --batch-size 4\
    --weight-decay 0.0005 --data-augmentation ssd --weights-backbone VGG16_Weights.IMAGENET1K_FEATURES
```

### SSDlite320 MobileNetV3-Large
```
torchrun --nproc_per_node=8 train.py\
    --dataset coco --model ssdlite320_mobilenet_v3_large --epochs 660\
    --aspect-ratio-group-factor 3 --lr-scheduler cosineannealinglr --lr 0.15 --batch-size 24\
    --weight-decay 0.00004 --data-augmentation ssdlite --weights-backbone MobileNet_V3_Large_Weights.IMAGENET1K_V1
```


### Mask R-CNN
```
torchrun --nproc_per_node=8 train.py\
    --dataset coco --model maskrcnn_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3 --weights-backbone ResNet50_Weights.IMAGENET1K_V1
```


### Keypoint R-CNN
```
torchrun --nproc_per_node=8 train.py\
    --dataset coco_kp --model keypointrcnn_resnet50_fpn --epochs 46\
    --lr-steps 36 43 --aspect-ratio-group-factor 3 --weights-backbone ResNet50_Weights.IMAGENET1K_V1
```


# torchvision fine-tuning manual

입력 

- bounding box 좌표는 좌 상단 (x1, y1) 우 하단 (x2, y2)로 표시
- keypoints는 [x, y, visibility]로 표시 / 가시성 항목은 없으면 0 있는데 안보이면 1 보이면 2
- pycocotools 라이브러리로 학습된 model을 평가한다.
  pip install pycocotools





#### 사전 작업

> kpt_oks_sigmas 를 수정해야 한다.
>
> - pycocotools/cocoeval.py 파일을 수정한다.
>   self.kpt_oks_sigmas 값을 수정하면 된다.
> - oks sigma는 0.26(코), 0.25(눈), 0.35(귀), 0.79(어깨), 0.72(팔꿈치), 0.62(손목), 1.07(엉덩이), 0.87(무릎), 0.89(발목)로 pycocotools에서 제공된다.
>   sigma ^ 2 = E [Euclidean distances ^ 2 / object scale ^ 2]
>   https://cocodataset.org/#keypoints-eval
> - 설정하는 방법은 있지만 대충 근사치를 구하거나 1을 입력하도록 하자.. 보통 얼굴쪽보다 몸쪽에서 더 큰 값이 나오는 경향이 있다.
> - coco_eval.py를 수정해서 바꿀 수도 있다.



> custom
>
> > train
> >
> > > images
> >
> > > annotations
>
> >test
> >
> >> images
> >
> >> annotations
>
> > result
> >
> > > weights



image와 annotation 파일은 이름이 같아야 하고, annotation 파일의 구성은 다음과 같다.

###### annotation_file.json

>{
>	"bboxes" : [[x11, y11, x12, y12], [x21, y21, x22, y22]]
>	"keypoints" : [[[kx1, ky1, v], [kx2, ky2, v]]]
>}

ex)

> {
> 	"bboxes": [[100, 100, 200, 200], [300, 300, 400, 400]],
> 	"keypoints": [[[153, 172, 2], [372, 349, 2]]]
> }

