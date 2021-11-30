from PIL import Image
from config import Cfg
from utils.YOLO_Predict import YOLO

while True:
    imgNum = input('Please input image number:')
    model_path = 'WeightsFile/Model/cv0.pth'
    cfg_path = 'WeightsFile/Config/cv0.cfg'
    img_path = 'VOCdevkit/VOC2007/JPEGImages/{}.jpg'.format(imgNum)
    yolo = YOLO(Cfg, model_path, cfg_path)
    image = Image.open(img_path)
    image, _ = yolo.detect_image(image, conf_thres=0.8, nms_thres=0.45)
    image.show()
