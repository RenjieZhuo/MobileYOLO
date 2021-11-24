from nets.CreateYOLO import CreateNetwork
from easydict import EasyDict
import torch
import numpy as np
from utils.utils import parse_module_defs


def CreateNet(cfgfile):
    model = CreateNetwork(ClassNum=Cfg.num_classes, cfgfile=cfgfile)
    _, _, Cfg.prune_idx = parse_module_defs(model.module_defs)
    return model


def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]


def CountNum(train_path, val_path):
    with open(train_path, 'r') as f:
        num_train = (len(f.readlines()))
    with open(val_path, 'r') as f:
        num_val = (len(f.readlines()))
    return num_val, num_train


Cfg = EasyDict()
Cfg.input_shape = (608, 608, 3)
Cfg.w = Cfg.input_shape[1]
Cfg.h = Cfg.input_shape[0]
Cfg.Cuda = True if torch.cuda.is_available() else False
Cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
Cfg.anchors_path = 'model_data/glass_anchors.txt'
Cfg.classes_path = 'model_data/glass_classes.txt'
Cfg.classes = get_classes(Cfg.classes_path)
Cfg.anchors = get_anchors(Cfg.anchors_path)
Cfg.num_classes = len(Cfg.classes)
Cfg.train_path = './VOCdevkit/VOC2007/ImageSets/Main/train.txt'
Cfg.val_path = './VOCdevkit/VOC2007/ImageSets/Main/val.txt'
Cfg.num_val, Cfg.num_train = CountNum(Cfg.train_path, Cfg.val_path)

Cfg.lr = 0.001
Cfg.Batch_size = 8
Cfg.Epoch = 150
Cfg.pruneLambda = 0.3
Cfg.isPruneTrain = 0
