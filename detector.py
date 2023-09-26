import cv2
import torch.cuda
import numpy as np

from yolov5.utils.torch_utils import select_device
from yolov5.models.experimental import attempt_load
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression, scale_boxes, Profile, LOGGER
from yolov5.utils.plots import colors
from yolov5.models.common import DetectMultiBackend


def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


class Detector(object):

    def __init__(self, weights):
        # params
        self.weights = weights
        self.imgsz = [640, 640]
        self.augment = False
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.classes = None
        self.agnostic_nms = False
        self.max_det = 1000
        self.line_thickness = 1
        self.half = False
        self.batch_size = 1
        # load model
        self.device = select_device('')
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=False, data=None, fp16=self.half)
        self.stride = self.model.stride
        self.names = self.model.names
        self.pt = self.model.pt
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else self.batch_size, 3, *self.imgsz))

    def detect(self, image):
        dt = (Profile(), Profile(), Profile())
        # precess
        with dt[0]:
            img0 = image.copy()
            img = letterbox(img0, self.imgsz, stride=self.stride)[0]
            img = img.transpose((2, 0, 1))[::-1]
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.model.device)
            img = img.half() if self.model.fp16 else img.float()
            img /= 255.0
            if len(img.shape) == 3:
                img = img[None]
        # run
        with dt[1]:
            pred = self.model(img, augment=self.augment)[0]
        with dt[2]:
            pred = non_max_suppression(pred,
                                       self.conf_thres,
                                       self.iou_thres,
                                       self.classes,
                                       self.agnostic_nms,
                                       max_det=self.max_det)
        alarm = False
        for i, det in enumerate(pred):
            if det is not None and len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], image.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f'{self.names[c]} {conf:.2f}'
                    plot_one_box(xyxy,
                                 image,
                                 label=label,
                                 color=colors(c, True),
                                 line_thickness=self.line_thickness)
                    if c == 0:
                        alarm = True
        LOGGER.info(f"infer:{dt[1].dt * 1E3:.1f}ms nms:{dt[2].dt * 1E3:.1f}ms")
        return alarm, image
