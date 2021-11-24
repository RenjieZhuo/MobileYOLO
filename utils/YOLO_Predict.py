import colorsys
import os

import numpy as np
import torch
from PIL import ImageFont, ImageDraw

from config import CreateNet
from utils.utils import non_max_suppression, DecodeBox, letterbox_image, yolo_correct_boxes


class YOLO(object):
    def __init__(self, Cfg, model_path, cfgfile):
        self.Cfg = Cfg
        self.anchors = self._get_anchors()
        self.generate(model_path, cfgfile)

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.Cfg.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape([-1, 3, 2])[::1, :, :]

    def generate(self, model_path, cfgfile):
        self.net = CreateNet(cfgfile)
        state_dict = torch.load(model_path, map_location=self.Cfg.device)
        self.net.load_state_dict(state_dict)
        if self.Cfg.Cuda:
            # self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
        self.yolo_decodes = []
        for i in range(3):
            self.yolo_decodes.append(
                DecodeBox(self.anchors[i], len(self.Cfg.classes),
                          (self.Cfg.input_shape[1], self.Cfg.input_shape[0])))
        hsv_tuples = [(x / len(self.Cfg.classes), 1., 1.)
                      for x in range(len(self.Cfg.classes))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    def detect_image(self, image, conf_thres=0.5, nms_thres=0.45):
        image_shape = np.array(np.shape(image)[0:2])
        crop_img = np.array(letterbox_image(image, (self.Cfg.input_shape[1], self.Cfg.input_shape[0])))
        photo = np.array(crop_img, dtype=np.float32)
        photo /= 255.0
        photo = np.transpose(photo, (2, 0, 1))
        photo = photo.astype(np.float32)
        images = []
        images.append(photo)
        images = np.asarray(images)
        with torch.no_grad():
            images = torch.from_numpy(images)
            if self.Cfg.Cuda:
                images = images.cuda()
            outputs = self.net(images)
        output_list = []
        for i in range(3):
            output_list.append(self.yolo_decodes[i](outputs[i]))
        output = torch.cat(output_list, 1)
        batch_detections = non_max_suppression(output, len(self.Cfg.classes),
                                               conf_thres=conf_thres,
                                               nms_thres=nms_thres)
        try:
            batch_detections = batch_detections[0].cpu().numpy()
        except:
            return image, False
        top_index = batch_detections[:, 4] * batch_detections[:, 5] > conf_thres
        top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
        top_label = np.array(batch_detections[top_index, -1], np.int32)
        top_bboxes = np.array(batch_detections[top_index, :4])
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(top_bboxes[:, 1],
                                                                                                      -1), np.expand_dims(
            top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)
        boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                   np.array([self.Cfg.input_shape[0], self.Cfg.input_shape[1]]), image_shape)
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = (np.shape(image)[0] + np.shape(image)[1]) // self.Cfg.input_shape[0]
        for i, c in enumerate(top_label):
            predicted_class = self.Cfg.classes[c]
            score = top_conf[i]
            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[self.Cfg.classes.index(predicted_class)], width=3)
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[self.Cfg.classes.index(predicted_class)])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image, True
