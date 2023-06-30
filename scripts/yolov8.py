import os
import rospy
import cv2

import numpy as np

from numpy import ndarray
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

from ultralytics.yolo.v8.detect import DetectionPredictor, DetectionTrainer, DetectionValidator

DATASETS = os.path.dirname(os.path.realpath(__file__)).replace('scripts', 'src/ultralytics/ultralytics/datasets/')

class DetectionPublisher:
    
    @staticmethod
    def train(model : str,
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
              name : str,
              weights : str = '' ) -> bool:
        
        """
        Initialize super DetectionTrainer and execute the train
        
        Params:
        model         -- path to model file, i.e. yolov8n.pt, yolov8n.yaml
        data          -- path to data file, i.e. coco128.yaml
        epochs        -- number of epochs to train for
        patience      -- epochs to wait for no observable improvement for early stopping of training
        batch         -- number of images per batch (-1 for AutoBatch)
        imgsz         -- size of input images as integer or w,h
        save          -- save train checkpoints and predict results
        save_period   -- Save checkpoint every x epochs (disabled if < 1)
        cache         -- True/ram, disk or False. Use cache for data loading
        device        -- device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
        workers       -- number of worker threads for data loading (per RANK if DDP)
        project       -- project name
        name          -- experiment name 

        weights    -- model to load if you'll use transferlearning

        Return:
        True
        """
        name= data.split('/')[0]
        data= DATASETS + data
        

        rospy.logwarn(f"data= {data}  name= {name}")

        args = dict(model=model, data=data, epochs=epochs, patience=patience, batch=batch, imgsz=imgsz, save=save, \
                    save_period=save_period, cache=cache, device=device, workers=workers, project=project, name=name)
        
        trainer = DetectionTrainer(overrides=args)
        if weights:
            trainer.model = trainer.get_model(cfg=model, weights=weights)

        trainer.train()
        return True
    
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
        


    

    

