import math

import numpy as np
import torch
import torch.nn as nn


class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size, cuda=True):
        super(YOLOLoss, self).__init__()
        self.anchors = np.reshape(anchors, [-1, 2])
        self.num_anchors = len(self.anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size
        self.feature_length = [img_size[0] // 32, img_size[0] // 16, img_size[0] // 8]
        self.label_smooth = 1
        self.ignore_threshold = 0.5
        self.cuda = cuda
        self.num = int(self.num_anchors / 3)

    def forward(self, input, targets=None):
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)

        stride_h = self.img_size[1] / in_h
        stride_w = self.img_size[0] / in_w

        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]

        prediction = input.view(bs, self.num, self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
        conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])
        mask, noobj_mask, t_box, tconf, tcls, box_loss_scale_x, box_loss_scale_y = self.get_target(targets,
                                                                                                   scaled_anchors, in_w,
                                                                                                   in_h)

        noobj_mask, pred_boxes_for_ciou = self.get_ignore(prediction, targets, scaled_anchors, in_w, in_h, noobj_mask)

        if self.cuda:
            mask, noobj_mask = mask.cuda(), noobj_mask.cuda()
            box_loss_scale_x, box_loss_scale_y = box_loss_scale_x.cuda(), box_loss_scale_y.cuda()
            tconf, tcls = tconf.cuda(), tcls.cuda()
            pred_boxes_for_ciou = pred_boxes_for_ciou.cuda()
            t_box = t_box.cuda()

        box_loss_scale = 2 - box_loss_scale_x * box_loss_scale_y
        ciou = (1 - self.box_ciou(pred_boxes_for_ciou[mask.bool()], t_box[mask.bool()])) * box_loss_scale[mask.bool()]
        loss_loc = torch.sum(ciou / bs)
        loss_conf = torch.sum(self.BCELoss(conf, mask) * mask / bs) + torch.sum(self.BCELoss(conf, mask) * noobj_mask / bs)
        loss_cls = torch.sum(self.BCELoss(pred_cls[mask == 1],self.smooth_labels(tcls[mask == 1], self.label_smooth, self.num_classes)) / bs)
        loss = loss_conf + loss_cls + loss_loc
        return loss, loss_conf.item(), loss_cls.item(), loss_loc.item()

    def get_target(self, target, anchors, in_w, in_h):
        bs = len(target)
        anchor_index = [[0, 1, 2], [3, 4, 5], [6, 7, 8]][self.feature_length.index(in_w)]
        subtract_index = [0, 3, 6][self.feature_length.index(in_w)]
        mask = torch.zeros(bs, self.num, in_h, in_w, requires_grad=False)
        noobj_mask = torch.ones(bs, self.num, in_h, in_w, requires_grad=False)
        tx = torch.zeros(bs, self.num, in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, self.num, in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, self.num, in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, self.num, in_h, in_w, requires_grad=False)
        t_box = torch.zeros(bs, self.num, in_h, in_w, 4, requires_grad=False)
        tconf = torch.zeros(bs, self.num, in_h, in_w, requires_grad=False)
        tcls = torch.zeros(bs, self.num, in_h, in_w, self.num_classes, requires_grad=False)
        box_loss_scale_x = torch.zeros(bs, self.num, in_h, in_w, requires_grad=False)
        box_loss_scale_y = torch.zeros(bs, self.num, in_h, in_w, requires_grad=False)
        for b in range(bs):
            if len(target[b]) == 0:
                continue
            gxs = target[b][:, 0:1] * in_w
            gys = target[b][:, 1:2] * in_h
            gws = target[b][:, 2:3] * in_w
            ghs = target[b][:, 3:4] * in_h
            gis = torch.floor(gxs)
            gjs = torch.floor(gys)
            gt_box = torch.FloatTensor(torch.cat([torch.zeros_like(gws), torch.zeros_like(ghs), gws, ghs], 1))
            anchor_shapes = torch.FloatTensor(
                torch.cat((torch.zeros((self.num_anchors, 2)), torch.FloatTensor(anchors)), 1))
            anch_ious = self.calculate_iou(gt_box, anchor_shapes)
            best_ns = torch.argmax(anch_ious, dim=-1)
            for i, best_n in enumerate(best_ns):
                if best_n not in anchor_index:
                    continue
                gi = gis[i].long()
                gj = gjs[i].long()
                gx = gxs[i]
                gy = gys[i]
                gw = gws[i]
                gh = ghs[i]
                if (gj < in_h) and (gi < in_w):
                    best_n = best_n - subtract_index
                    noobj_mask[b, best_n, gj, gi] = 0
                    mask[b, best_n, gj, gi] = 1
                    tx[b, best_n, gj, gi] = gx
                    ty[b, best_n, gj, gi] = gy
                    tw[b, best_n, gj, gi] = gw
                    th[b, best_n, gj, gi] = gh
                    box_loss_scale_x[b, best_n, gj, gi] = target[b][i, 2]
                    box_loss_scale_y[b, best_n, gj, gi] = target[b][i, 3]
                    tconf[b, best_n, gj, gi] = 1
                    tcls[b, best_n, gj, gi, target[b][i, 4].long()] = 1
                else:
                    continue
        t_box[..., 0] = tx
        t_box[..., 1] = ty
        t_box[..., 2] = tw
        t_box[..., 3] = th
        return mask, noobj_mask, t_box, tconf, tcls, box_loss_scale_x, box_loss_scale_y

    def get_ignore(self, prediction, target, scaled_anchors, in_w, in_h, noobj_mask):
        bs = len(target)
        anchor_index = [[0, 1, 2], [3, 4, 5], [6, 7, 8]][self.feature_length.index(in_w)]
        scaled_anchors = np.array(scaled_anchors)[anchor_index]
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(int(bs * self.num), 1, 1).view(x.shape).type(
            FloatTensor)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(int(bs * self.num), 1, 1).view(
            y.shape).type(FloatTensor)
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))

        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x + grid_x
        pred_boxes[..., 1] = y + grid_y
        pred_boxes[..., 2] = torch.exp(w) * anchor_w
        pred_boxes[..., 3] = torch.exp(h) * anchor_h
        for i in range(bs):
            pred_boxes_for_ignore = pred_boxes[i]
            pred_boxes_for_ignore = pred_boxes_for_ignore.view(-1, 4)
            if len(target[i]) > 0:
                gx = target[i][:, 0:1] * in_w
                gy = target[i][:, 1:2] * in_h
                gw = target[i][:, 2:3] * in_w
                gh = target[i][:, 3:4] * in_h
                gt_box = torch.FloatTensor(torch.cat([gx, gy, gw, gh], -1)).type(FloatTensor)
                anch_ious = self.calculate_iou(gt_box, pred_boxes_for_ignore)
                anch_ious_max, _ = torch.max(anch_ious, dim=0)
                anch_ious_max = anch_ious_max.view(pred_boxes[i].size()[:3])
                noobj_mask[i][anch_ious_max > self.ignore_threshold] = 0
        return noobj_mask, pred_boxes

    def calculate_iou(self, _box_a, _box_b):
        b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
        b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
        b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
        b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2
        box_a = torch.zeros_like(_box_a)
        box_b = torch.zeros_like(_box_b)
        box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
        box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2
        A = box_a.size(0)
        B = box_b.size(0)
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                           box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                           box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        inter = inter[:, :, 0] * inter[:, :, 1]
        area_a = ((box_a[:, 2] - box_a[:, 0]) *
                  (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
        area_b = ((box_b[:, 2] - box_b[:, 0]) *
                  (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
        union = area_a + area_b - inter
        return inter / union

    def smooth_labels(self, y_true, label_smoothing, num_classes):
        return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

    def box_ciou(self, b1, b2):
        b1_xy = b1[..., :2]
        b1_wh = b1[..., 2:4]
        b1_wh_half = b1_wh / 2.
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half
        b2_xy = b2[..., :2]
        b2_wh = b2[..., 2:4]
        b2_wh_half = b2_wh / 2.
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half
        intersect_mins = torch.max(b1_mins, b2_mins)
        intersect_maxes = torch.min(b1_maxes, b2_maxes)
        intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        union_area = b1_area + b2_area - intersect_area
        iou = intersect_area / torch.clamp(union_area, min=1e-6)
        center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)
        enclose_mins = torch.min(b1_mins, b2_mins)
        enclose_maxes = torch.max(b1_maxes, b2_maxes)
        enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
        enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), axis=-1)
        ciou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal, min=1e-6)
        v = (4 / (math.pi ** 2)) * torch.pow(
            (torch.atan(b1_wh[..., 0] / torch.clamp(b1_wh[..., 1], min=1e-6)) - torch.atan(
                b2_wh[..., 0] / torch.clamp(b2_wh[..., 1], min=1e-6))), 2)
        alpha = v / torch.clamp((1.0 - iou + v), min=1e-6)
        ciou = ciou - alpha * v
        return ciou

    def MSELoss(self, pred, target):
        return (pred - target) ** 2

    def BCELoss(self, pred, target):
        epsilon = 1e-7
        t = pred.float()
        result = (t >= epsilon).float() * t + (t < epsilon).float() * epsilon
        pred = (result <= (1.0 - epsilon)).float() * result + (result > (1.0 - epsilon)).float() * (1.0 - epsilon)
        output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output
