# MobileYOLO
## BrokenGlass: A Benchmark Dataset for Glass Curtain Wall Automatic Inspection by UAVs
submitted to CVPR2022

### Directory structure
```
├── Networks                              
│   ├── CreateYOLO.py         
│   ├── IRB_yolov4.cfg    
│   └── yolov4.cfg
├── utils
│   ├── dataloader.py       
│   ├── utils.py 
│   ├── yolo_loss.py
│   ├── YOLO_Predict.py
│   ├── glass_anchors.txt
│   ├── glass_classes.txt
│   └── simhei.ttf
├── WeightsFile
│   ├── IRBYOLO_SparseTraining     
│   └── MobileYOLO
│       ├── Config
│       └── Model
├── VOCdevkit                  
│   └── VOC2007
│       ├── Annotations                
│       ├── JPEGImages
│       └── ImageSets
│           └── Main
├── config.py
├── metrics.py         
├── predict.py             
├── PruneModel.py
├── README.md
└── train.py                      
```

### Usage
#### Normal training
1. set Cfg.isPruneTrain = 0 in config.py
2. set cfgfile, model_path_train in train.py
3. run train.py

#### Sparsity training
1. set Cfg.isPruneTrain = 1 in config.py
2. set Cfg.pruneLambda in config.py
3. set cfgfile, model_path_train in train.py
4. run train.py

#### Channel pruning
1. set cfgfile, modelfile,prune_percent in PruneModel.py
2. run PruneModel.py

#### Predict
1. set cfg_path, model_path in predict.py
2. run predict.py
3. input image number

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
