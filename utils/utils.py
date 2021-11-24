from __future__ import division
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.ops import nms


class DecodeBox(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(DecodeBox, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size

    def forward(self, input):
        batch_size = input.size(0)
        input_height = input.size(2)
        input_width = input.size(3)
        stride_h = self.img_size[1] / input_height
        stride_w = self.img_size[0] / input_width
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in
                          self.anchors]

        prediction = input.view(batch_size, self.num_anchors,
                                self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])

        w = prediction[..., 2]
        h = prediction[..., 3]

        conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
            batch_size * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
            batch_size * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)

        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) * _scale,
                            conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
        return output.data


def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def yolo_correct_boxes(top, left, bottom, right, input_shape, image_shape):
    new_shape = image_shape * np.min(input_shape / image_shape)

    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    box_yx = np.concatenate(((top + bottom) / 2, (left + right) / 2), axis=-1) / input_shape
    box_hw = np.concatenate((bottom - top, right - left), axis=-1) / input_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
        conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]
        if not image_pred.size(0):
            continue
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

        unique_labels = detections[:, -1].cpu().unique()

        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
            detections = detections.cuda()

        for c in unique_labels:
            detections_class = detections[detections[:, -1] == c]
            keep = nms(
                detections_class[:, :4],
                detections_class[:, 4]*detections_class[:, 5],
                nms_thres
            )
            max_detections = detections_class[keep]
            output[image_i] = max_detections if output[image_i] is None else torch.cat(
                (output[image_i], max_detections))

    return output


def bbox_iou(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou


def parse_module_defs(module_defs):
    CBL_idx = []
    Conv_idx = []
    ignore_idx = set()
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == '1':
                CBL_idx.append(i)
            else:
                Conv_idx.append(i)
            if module_defs[i + 1]['type'] == 'maxpool' and module_defs[i + 2]['type'] == 'route':
                ignore_idx.add(i)
            if module_defs[i + 1]['type'] == 'yolohead':
                ignore_idx.add(i)
        elif module_def['type'] == 'shortcut':
            ignore_idx.add(i - 1)
            identity_idx = (i + int(module_def['from']))
            if module_defs[identity_idx]['type'] == 'convolutional':
                ignore_idx.add(identity_idx)
            elif module_defs[identity_idx]['type'] == 'shortcut':
                ignore_idx.add(identity_idx - 1)
        elif module_def['type'] == 'upsample':
            ignore_idx.add(i - 1)
    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]
    return CBL_idx, Conv_idx, prune_idx


def parse_model_cfg(path):
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]
    mdefs = []
    for line in lines:
        if line.startswith('['):
            mdefs.append({})
            mdefs[-1]['type'] = line[1:-1].rstrip()
            if mdefs[-1]['type'] == 'convolutional':
                mdefs[-1]['batch_normalize'] = 0
        else:
            key, val = line.split("=")
            key = key.rstrip()

            if 'anchors' in key:
                mdefs[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))
            else:
                mdefs[-1][key] = val.strip()
    return mdefs


def parse_module_defs2(module_defs):
    CBL_idx = []
    Conv_idx = []
    shortcut_idx = dict()
    shortcut_all = set()
    ignore_idx = set()
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == '1':
                CBL_idx.append(i)
            else:
                Conv_idx.append(i)
            if module_defs[i + 1]['type'] == 'maxpool' and module_defs[i + 2]['type'] == 'route':
                ignore_idx.add(i)
            if module_defs[i + 1]['type'] == 'yolohead':
                ignore_idx.add(i)
        elif module_def['type'] == 'upsample':
            ignore_idx.add(i - 1)
        elif module_def['type'] == 'shortcut':
            identity_idx = (i + int(module_def['from']))
            if module_defs[identity_idx]['type'] == 'convolutional':
                shortcut_idx[i - 1] = identity_idx
                shortcut_all.add(identity_idx)
            elif module_defs[identity_idx]['type'] == 'shortcut':
                shortcut_idx[i - 1] = identity_idx - 1
                shortcut_all.add(identity_idx - 1)
            shortcut_all.add(i - 1)
    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]
    return CBL_idx, Conv_idx, prune_idx, shortcut_idx, shortcut_all


def gather_bn_weights(module_list, prune_idx):
    size_list = [module_list[idx][1].weight.data.shape[0] for idx in prune_idx]
    bn_weights = torch.zeros(sum(size_list))
    index = 0
    for idx, size in zip(prune_idx, size_list):
        bn_weights[index:(index + size)] = module_list[idx][1].weight.data.abs().clone()
        index += size

    return bn_weights


def merge_mask(model, CBLidx2mask, CBLidx2filters):
    for i in range(len(model.module_defs) - 1, -1, -1):
        mtype = model.module_defs[i]['type']
        if mtype == 'shortcut':
            if model.module_defs[i]['is_access']:
                continue
            Merge_masks = []
            layer_i = i
            while mtype == 'shortcut':
                model.module_defs[layer_i]['is_access'] = True
                if model.module_defs[layer_i - 1]['type'] == 'convolutional':
                    bn = int(model.module_defs[layer_i - 1]['batch_normalize'])
                    if bn:
                        Merge_masks.append(CBLidx2mask[layer_i - 1].unsqueeze(0))
                layer_i = int(model.module_defs[layer_i]['from']) + layer_i
                mtype = model.module_defs[layer_i]['type']
                if mtype == 'convolutional':
                    bn = int(model.module_defs[layer_i]['batch_normalize'])
                    if bn:
                        Merge_masks.append(CBLidx2mask[layer_i].unsqueeze(0))
            if len(Merge_masks) > 1:
                Merge_masks = torch.cat(Merge_masks, 0)
                merge_mask = (torch.sum(Merge_masks, dim=0) > 0).float()
            else:
                merge_mask = Merge_masks[0].float()
            layer_i = i
            mtype = 'shortcut'
            while mtype == 'shortcut':
                if model.module_defs[layer_i - 1]['type'] == 'convolutional':
                    bn = int(model.module_defs[layer_i - 1]['batch_normalize'])
                    if bn:
                        CBLidx2mask[layer_i - 1] = merge_mask
                        CBLidx2filters[layer_i - 1] = int(torch.sum(merge_mask).item())
                layer_i = int(model.module_defs[layer_i]['from']) + layer_i
                mtype = model.module_defs[layer_i]['type']
                if mtype == 'convolutional':
                    bn = int(model.module_defs[layer_i]['batch_normalize'])
                    if bn:
                        CBLidx2mask[layer_i] = merge_mask
                        CBLidx2filters[layer_i] = int(torch.sum(merge_mask).item())

    for i in range(len(model.module_defs)):
        mtype = model.module_defs[i]['type']
        if mtype == 'convolutional' and model.module_defs[i]['group'] != '1':
            Merge_masks = []
            layer_i = i
            if model.module_defs[layer_i - 1]['type'] == 'convolutional':
                Merge_masks.append(CBLidx2mask[layer_i - 1].unsqueeze(0))
            elif model.module_defs[i - 1]['type'] == 'route':
                layer_i = int(model.module_defs[i - 1]['layers']) + i - 1
                Merge_masks.append(CBLidx2mask[layer_i].unsqueeze(0))
            Merge_masks.append(CBLidx2mask[i].unsqueeze(0))
            Merge_masks = torch.cat(Merge_masks, dim=0)
            merge_mask = (torch.sum(Merge_masks, dim=0) > 0).float()
            CBLidx2mask[i] = merge_mask
            CBLidx2filters[i] = int(torch.sum(merge_mask).item())
            layer_i = i
            if model.module_defs[layer_i - 1]['type'] == 'convolutional':
                CBLidx2mask[layer_i - 1] = merge_mask
                CBLidx2filters[layer_i - 1] = int(torch.sum(merge_mask).item())
            elif model.module_defs[i - 1]['type'] == 'route':
                layer_i = int(model.module_defs[i - 1]['layers']) + i - 1
                CBLidx2mask[layer_i] = merge_mask
                CBLidx2filters[layer_i] = int(torch.sum(merge_mask).item())
                CBLidx2mask[layer_i + 1] = merge_mask
                CBLidx2filters[layer_i + 1] = int(torch.sum(merge_mask).item())


def get_input_mask(module_defs, idx, CBLidx2mask):
    if idx == 0:
        return np.ones(3)
    if module_defs[idx - 1]['type'] == 'convolutional':
        return CBLidx2mask[idx - 1]
    elif module_defs[idx - 1]['type'] == 'shortcut':
        return CBLidx2mask[idx - 2]
    elif module_defs[idx - 1]['type'] == 'route':
        route_in_idxs = []
        for layer_i in module_defs[idx - 1]['layers'].split(","):
            if int(layer_i) < 0:
                route_in_idxs.append(idx - 1 + int(layer_i))
            else:
                route_in_idxs.append(int(layer_i))
        if len(route_in_idxs) == 1:
            if module_defs[route_in_idxs[0]]['type'] == 'shortcut':
                mask = CBLidx2mask[route_in_idxs[0] - 1]
            else:
                mask = CBLidx2mask[route_in_idxs[0]]
            return mask

        elif len(route_in_idxs) == 2:
            if module_defs[route_in_idxs[0]]['type'] == 'upsample':
                mask1 = CBLidx2mask[route_in_idxs[0] - 1]
            elif module_defs[route_in_idxs[0]]['type'] == 'convolutional':
                mask1 = CBLidx2mask[route_in_idxs[0]]
            if module_defs[route_in_idxs[1]]['type'] == 'convolutional':
                mask2 = CBLidx2mask[route_in_idxs[1]]
            else:
                mask2 = CBLidx2mask[route_in_idxs[1] - 1]
            return np.concatenate([mask1, mask2])

        elif len(route_in_idxs) == 4:
            mask = CBLidx2mask[route_in_idxs[-1]]
            return np.concatenate([mask, mask, mask, mask])
        else:
            print("Something wrong with route module!")
            raise Exception
    elif module_defs[idx - 1]['type'] == 'maxpool':
        if module_defs[idx - 2]['type'] == 'route':
            return get_input_mask(module_defs, idx - 1, CBLidx2mask)
        else:
            return CBLidx2mask[idx - 2]


def init_weights_from_loose_model(compact_model, loose_model, CBL_idx, Conv_idx, CBLidx2mask):
    for idx in CBL_idx:
        compact_CBL = compact_model.module_list[idx]
        loose_CBL = loose_model.module_list[idx]
        out_channel_idx = np.argwhere(CBLidx2mask[idx])[:, 0].tolist()
        compact_bn, loose_bn = compact_CBL[1], loose_CBL[1]
        compact_bn.weight.data = loose_bn.weight.data[out_channel_idx].clone()
        compact_bn.bias.data = loose_bn.bias.data[out_channel_idx].clone()
        compact_bn.running_mean.data = loose_bn.running_mean.data[out_channel_idx].clone()
        compact_bn.running_var.data = loose_bn.running_var.data[out_channel_idx].clone()
        input_mask = get_input_mask(loose_model.module_defs, idx, CBLidx2mask)
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
        compact_conv, loose_conv = compact_CBL[0], loose_CBL[0]
        if loose_conv.groups == 1:
            tmp = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
            compact_conv.weight.data = tmp[out_channel_idx, :, :, :].clone()
        else:
            tmp = loose_conv.weight.data[in_channel_idx, :, :, :].clone()
            compact_conv.weight.data = tmp.clone()
    for idx in Conv_idx:
        compact_conv = compact_model.module_list[idx][0]
        loose_conv = loose_model.module_list[idx][0]
        input_mask = get_input_mask(loose_model.module_defs, idx, CBLidx2mask)
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
        compact_conv.weight.data = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.bias.data = loose_conv.bias.data.clone()


def write_cfg(cfg_file, module_defs):
    with open(cfg_file, 'w') as f:
        for module_def in module_defs:
            f.write(f"[{module_def['type']}]\n")
            for key, value in module_def.items():
                if key == 'batch_normalize' and value == 0:
                    continue
                if key != 'type':
                    if key == 'anchors':
                        value = ', '.join(','.join(str(int(i)) for i in j) for j in value)
                    f.write(f"{key}={value}\n")
            f.write("\n")
    return cfg_file


def convert_annotation(image_id, file, classes):
    file.write('VOCdevkit/VOC2007/JPEGImages/{}.jpg'.format(image_id))
    in_file = open('VOCdevkit/VOC2007/Annotations/{}.xml'.format(image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


def Generate_Train(train_path, val_path, valFlag, classes):
    with open(val_path, 'w') as fval:
        with open('./VOCdevkit/VOC2007/ImageSets/Main/CrossVal_{}.txt'.format(valFlag), 'r') as cvval:
            listval = cvval.readlines()
        for item in listval:
            convert_annotation(item.strip(), fval, classes)
            fval.write('\n')

    with open(train_path, 'w') as ftrain:
        for i in range(5):
            if i != valFlag:
                with open('./VOCdevkit/VOC2007/ImageSets/Main/CrossVal_{}.txt'.format(i), 'r') as cvtrain:
                    listtrain = cvtrain.readlines()
                for item in listtrain:
                    convert_annotation(item.strip(), ftrain, classes)
                    ftrain.write('\n')
