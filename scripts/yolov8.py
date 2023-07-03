import os
import rospy
import cv2
from src.ultralytics.ultralytics.yolo.engine.results import Results
import torch
import numpy as np

from numpy import ndarray
from typing import Tuple, Union, List
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

from ultralytics.yolo.v8.detect import DetectionPredictor, DetectionTrainer, DetectionValidator


PATH = os.path.dirname(os.path.realpath(__file__))
DATASETS = PATH.replace('scripts', 'src/ultralytics/ultralytics/datasets/')
DETECTION = PATH.replace('scripts', 'src/ultralytics/ultralytics/yolo/v8/detect/')


class Detection:
    def __init__(self,
                model : str,
                data : str, 
                epochs : int, 
                patience: int, 
                batch : int, 
                imgsz : int, 
                save : bool, 
                save_period : int,
                cache : bool,
                device : int,
                workers : int,
                project : str,
                name : str) -> bool:
        
        """
        Initialize super DetectionTrainer and execute the train\n
        
        Params:\n
        model         -- path to model file, i.e. yolov8n.pt, yolov8n.yaml\n
        data          -- path to data file, i.e. coco128.yaml\n
        epochs        -- number of epochs to train for\n
        patience      -- epochs to wait for no observable improvement for early stopping of training\n
        batch         -- number of images per batch (-1 for AutoBatch)\n
        imgsz         -- size of input images as integer or w,h\n
        save          -- save train checkpoints and predict results\n
        save_period   -- Save checkpoint every x epochs (disabled if < 1)\n
        cache         -- True/ram, disk or False. Use cache for data loading\n
        device        -- device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu\n
        workers       -- number of worker threads for data loading (per RANK if DDP)\n
        project       -- project name\n
        name          -- experiment name \n

        Return:\n
        
        """
        name= data.split('/')[0]
        data= DATASETS + data
        project = DETECTION
        self._args = dict(model=model, data=data, epochs=epochs, patience=patience, batch=batch, imgsz=imgsz, save=save, \
                    save_period=save_period, cache=cache, device=device, workers=workers, project=project, name=name)
        
        self.__trainer = None
        self.__predictor = None

    @property
    def args(self) -> dict:
        return self._args
    
    @args.setter
    def args(self, _args : dict):
          self.__set_args(_args)

    
    def _set_args(self, args : dict):
        """
        Change inital params to execute train, inference or validation\n

        Params: \n
        args         -- arguments to be changed or added in inital params\n

        Returns:\n

        """
        for arg in args:
            self._args[arg] = args[arg]  

    def set_trainer(self, args : dict = None):
        self.__trainer = DetectionTrainer(overrides=args) if args else DetectionTrainer(overrides=self._args)

    def train(self, model : str, data : str, weights : str = '', device : str = '0'):
        """
        Executte the train

        Params: \n
        model         -- path to model file, i.e. yolov8n.pt, yolov8n.yaml\n
        data          -- path to data file, i.e. coco128.yaml\n
        weights       -- model to load if you'll use transferlearning\n
        device        -- device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu\n

        Returns: \n

        """
        data = DATASETS + data
        device = int(device) if device != 'cpu' else device
        self._set_args(dict(model=model, data=data, device=device))

        if weights:
            self.__trainer.model = self.__trainer.get_model(cfg=model, weights=weights)
        
        self.__trainer.train()

    def set_predictor(self, args : dict):
        self._set_args(args)
        self.__predictor = DetectionPredictor(overrides=self._args)



    @torch.no_grad()
    def _inference(self, img: torch.Tensor) -> List[Results]:
        """
        Executte de inference on input image\n

        Params:\n
        img         -- tensor [c, h, w]\n

        Returns:\n
        detections  -- tensor of shape [num_boxes, 6], where each item is represented as [x1, y1, x2, y2, confidence, class_id]\n
        """
        img = img.unsqueeze(0)
        detections = self.__predictor.stream_inference(source=img, model=self._args['model'])
        return detections


class DetectionPublisher:
    
    @staticmethod
    def ros_to_cv2(img_msg : Image) -> ndarray:
        """
        Convert a ros_msg in opencv format

        Params:
        img_msg     -- image in ros_msg format

        Returns:
        img_cv      -- image in opencv format
        """
        try:
            img_cv = CvBridge().imgmsg_to_cv2(img_msg, "passthrough")
            return img_cv

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    
    @staticmethod
    def cv2_to_ros(img_msg : ndarray) -> Image:
        """
        Convert a opencv in ros_msg format

        Params:
        img_msg     -- image in opencv format

        Returns:
        img_cv      -- image in ros_msg format
        """
        try:
            img_ros = CvBridge().cv2_to_imgmsg(img_msg, "passthrough")
            return img_ros

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def __init__(self, cam_topic : str):
        rospy.Subscriber(cam_topic, Image, self._call_cam)

        self.__pub_inference_view = rospy.Publisher("yolov8/inference_view", Image, queue_size=10)
        self.__pub_inference_result = rospy.Publisher("yolov8/inference_view", Detection2DArray, queue_size=10)

        self.__detector = DetectionPredictor()

    
    
    
    def _call_cam(self, img_msg : Image) -> Image:

        np_img_orig = self.ros_to_cv2(img_msg)
        # handle possible different img formats
        if len(np_img_orig.shape) == 2:
            np_img_orig = np.stack([np_img_orig] * 3, axis=2)

        h_orig, w_orig, c = np_img_orig.shape

        # automatically resize the image to the next smaller possible size
        w_scaled, h_scaled = self.img_size
        np_img_resized = cv2.resize(np_img_orig, (w_scaled, h_scaled))
        # conversion to torch tensor (copied from original yolov7 repo)
        img = np_img_resized.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = torch.from_numpy(np.ascontiguousarray(img))
        img = img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.
        img = img.to(self.device)
        


    

    

