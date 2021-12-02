# MobileYOLO
## BrokenGlass: A Benchmark Dataset for Glass Curtain Wall Automatic Inspection by UAVs
submitted to CVPR2022

### Directory structure
```
├── Networks                              
│   ├── CreateYOLO.py                      Build the network
│   ├── IRB_yolov4.cfg                     Network profile(IRB-YOLO)
│   └── yolov4.cfg                         Network profile(YOLOv4)
├── utils
│   ├── dataloader.py                      Dataloader
│   ├── utils.py 
│   ├── yolo_loss.py                       YOLO Loss
│   ├── YOLO_Predict.py
│   ├── glass_anchors.txt
│   ├── glass_classes.txt                  
│   └── simhei.ttf                         Font profile
├── WeightsFile                            Pre-trained models
│   ├── IRBYOLO_SparseTraining             Sparse Training models(IRB-YOLO)
│   └── MobileYOLO                         MobileYOLO
│       ├── Config
│       └── Model
├── VOCdevkit                             
│   └── VOC2007 
│       ├── Annotations                    Label(.xml)
│       ├── JPEGImages                     Images
│       └── ImageSets                      Data set division
│           └── Main
├── config.py                              Configuration file
├── metrics.py                             Compute mAP
├── predict.py                             Predictive images
├── PruneModel.py                          Pruning the model
├── README.md
└── train.py                               Training the model
```

### Usage
#### Normal training
1. Set Cfg.isPruneTrain = 0 in config.py
2. Update cfgfile, model_path_train in train.py
3. Run train.py

#### Sparsity training
1. Set Cfg.isPruneTrain = 1 in config.py
2. Modify Cfg.pruneLambda in config.py
3. Update cfgfile, model_path_train in train.py
4. Run train.py

#### Channel pruning
1. Set cfgfile, modelfile,prune_percent in PruneModel.py
2. Run PruneModel.py

#### Predict
1. Set cfg_path, model_path in predict.py
2. Run predict.py
3. Input image number

### Dataset
Dataset can be obtained via e-mail request!

### Requirements
* PyTorch == 1.9.0
* TorchVision == 0.10.0
* CUDA 11.0  cudnn 7.6.5

### Contact
E-Mail: zhuorenjie@foxmail.com

### Reference
https://github.com/bubbliiiing/yolov4-pytorch \
https://github.com/tanluren/yolov3-channel-and-layer-pruning
