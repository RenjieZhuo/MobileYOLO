import colorsys
import glob
import json
import os
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from config import Cfg
from utils.utils import non_max_suppression, DecodeBox, letterbox_image, yolo_correct_boxes


class mAP_Yolo(object):
    def __init__(self, net, **kwargs):
        self.cuda = Cfg.Cuda
        self.model_image_size = Cfg.input_shape
        self.class_names = Cfg.classes
        self.anchors = self._get_anchors()
        self.generate(net)

    def _get_anchors(self):
        anchors_path = os.path.expanduser(Cfg.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape([-1, 3, 2])[::1, :, :]

    def generate(self, net):
        self.net = net.eval()
        self.yolo_decodes = []
        for i in range(3):
            self.yolo_decodes.append(
                DecodeBox(self.anchors[i], len(self.class_names), (self.model_image_size[1], self.model_image_size[0])))

        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

    def detect_image(self, image_id, image):
        confidence = 0.01
        iou = 0.5
        f = open("./AP/detection-results/" + image_id + ".txt", "w")
        image_shape = np.array(np.shape(image)[0:2])

        crop_img = np.array(letterbox_image(image, (self.model_image_size[1], self.model_image_size[0])))
        photo = np.array(crop_img, dtype=np.float32)
        photo /= 255.0
        photo = np.transpose(photo, (2, 0, 1))
        photo = photo.astype(np.float32)
        images = []
        images.append(photo)
        images = np.asarray(images)

        with torch.no_grad():
            images = torch.from_numpy(images)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)

        output_list = []
        for i in range(3):
            output_list.append(self.yolo_decodes[i](outputs[i]))
        output = torch.cat(output_list, 1)
        batch_detections = non_max_suppression(output, len(self.class_names),
                                               conf_thres=confidence,
                                               nms_thres=iou)

        try:
            batch_detections = batch_detections[0].cpu().numpy()
        except:
            return image

        top_index = batch_detections[:, 4] * batch_detections[:, 5] > confidence
        top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
        top_label = np.array(batch_detections[top_index, -1], np.int32)
        top_bboxes = np.array(batch_detections[top_index, :4])
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(top_bboxes[:, 1],
                                                                                                      -1), np.expand_dims(
            top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)

        boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                   np.array([self.model_image_size[0], self.model_image_size[1]]), image_shape)

        for i, c in enumerate(top_label):
            predicted_class = self.class_names[c]
            score = str(top_conf[i])

            top, left, bottom, right = boxes[i]

            if np.isnan(top) or np.isnan(left) or np.isnan(bottom) or np.isnan(right):
                continue

            f.write("{} {} {} {} {} {}\n".format(
                predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return


def get_predict_txt(image_ids, yolo):
    if not os.path.exists("./AP"):
        os.makedirs("./AP")
    if not os.path.exists("./AP/detection-results"):
        os.makedirs("./AP/detection-results")

    for image_id in tqdm(image_ids):
        image_path = "VOCdevkit/VOC2007/JPEGImages/" + image_id + ".jpg"
        image = Image.open(image_path)
        yolo.detect_image(image_id, image)


def get_gt_txt(image_ids):
    if not os.path.exists("./AP/ground-truth"):
        os.makedirs("./AP/ground-truth")
        for image_id in image_ids:
            with open("./AP/ground-truth/" + image_id + ".txt", "w") as new_f:
                root = ET.parse("VOCdevkit/VOC2007/Annotations/" + image_id + ".xml").getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult') != None:
                        difficult = obj.find('difficult').text
                        if int(difficult) == 1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    bndbox = obj.find('bndbox')
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text
                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))


def get_AP(net):
    if os.path.exists("./AP"):
        shutil.rmtree("./AP")
    yolo = mAP_Yolo(net)
    image_ids = []
    f_val_image_ids = open(Cfg.val_path, 'r')
    line = f_val_image_ids.readline()
    while line:
        id = os.path.basename(line.split(' ')[0]).split('.')[0]
        image_ids.append(id)
        line = f_val_image_ids.readline()

    get_predict_txt(image_ids, yolo)

    get_gt_txt(image_ids)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    GT_PATH = os.path.join(os.getcwd(), 'AP', 'ground-truth')
    DR_PATH = os.path.join(os.getcwd(), 'AP', 'detection-results')

    def voc_ap(rec, prec):
        rec.insert(0, 0.0)
        rec.append(1.0)
        mrec = rec[:]
        prec.insert(0, 0.0)
        prec.append(0.0)
        mpre = prec[:]
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i - 1]:
                i_list.append(i)
        ap = 0.0
        for i in i_list:
            ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
        return ap, mrec, mpre

    def file_lines_to_list(path):
        with open(path) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        return content

    TEMP_FILES_PATH = ".temp_files"
    if not os.path.exists(TEMP_FILES_PATH):
        os.makedirs(TEMP_FILES_PATH)
    ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
    ground_truth_files_list.sort()
    gt_counter_per_class = {}
    counter_images_per_class = {}
    for txt_file in ground_truth_files_list:
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        lines_list = file_lines_to_list(txt_file)
        bounding_boxes = []
        is_difficult = False
        already_seen_classes = []
        for line in lines_list:
            if "difficult" in line:
                class_name, left, top, right, bottom, _difficult = line.split()
                is_difficult = True
            else:
                class_name, left, top, right, bottom = line.split()
            bbox = left + " " + top + " " + right + " " + bottom
            if is_difficult:
                bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False, "difficult": True})
                is_difficult = False
            else:
                bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    gt_counter_per_class[class_name] = 1

                if class_name not in already_seen_classes:
                    if class_name in counter_images_per_class:
                        counter_images_per_class[class_name] += 1
                    else:
                        counter_images_per_class[class_name] = 1
                    already_seen_classes.append(class_name)

        with open(TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    gt_classes = list(gt_counter_per_class.keys())
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)
    dr_files_list = glob.glob(DR_PATH + '/*.txt')
    dr_files_list.sort()

    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []
        for txt_file in dr_files_list:
            file_id = txt_file.split(".txt", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            lines = file_lines_to_list(txt_file)
            for line in lines:
                tmp_class_name, confidence, left, top, right, bottom = line.split()
                if tmp_class_name == class_name:
                    bbox = left + " " + top + " " + right + " " + bottom
                    bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})
        bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
        with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)
    sum_AP = 0.0
    count_true_positives = {}
    for class_index, class_name in enumerate(gt_classes):
        count_true_positives[class_name] = 0
        dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
        dr_data = json.load(open(dr_file))
        nd = len(dr_data)
        tp = [0] * nd
        fp = [0] * nd
        score = [0] * nd
        for idx, detection in enumerate(dr_data):
            file_id = detection["file_id"]
            score[idx] = float(detection["confidence"])
            gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
            ground_truth_data = json.load(open(gt_file))
            ovmax = -1
            gt_match = -1
            bb = [float(x) for x in detection["bbox"].split()]
            for obj in ground_truth_data:
                if obj["class_name"] == class_name:
                    bbgt = [float(x) for x in obj["bbox"].split()]
                    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                                          + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj
            min_overlap = 0.5
            if ovmax >= min_overlap:
                if "difficult" not in gt_match:
                    if not bool(gt_match["used"]):
                        tp[idx] = 1
                        gt_match["used"] = True
                        count_true_positives[class_name] += 1
                        with open(gt_file, 'w') as f:
                            f.write(json.dumps(ground_truth_data))
                    else:
                        fp[idx] = 1
            else:
                fp[idx] = 1
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
        ap, mrec, mprec = voc_ap(rec[:], prec[:])
        sum_AP += ap
    mAP = sum_AP / n_classes
    if os.path.exists("./.temp_files"):
        shutil.rmtree("./.temp_files")
    if os.path.exists("./AP"):
        shutil.rmtree("./AP")
    return mAP
