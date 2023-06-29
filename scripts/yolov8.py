import os
import rospy

from ultralytics.yolo.v8.detect import DetectionPredictor, DetectionTrainer, DetectionValidator

DIR = os.path.dirname(os.path.realpath(__file__)).replace('scripts', 'src/ultralytics/ultralytics/datasets/')

class Detection:#(DetectionPredictor, DetectionTrainer, DetectionValidator):
    
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
        load_model    -- Model to load if you'll use transferlearning

        Return:
        True
        """
        rospy.logwarn(DIR+data)
        args = dict(model=model, data=DIR+data, epochs=epochs, patience=patience, batch=batch, imgsz=imgsz, save=save, \
                    save_period=save_period, cache=cache, device=device)
        
        trainer = DetectionTrainer(overrides=args)
        if weights:
            trainer.model = trainer.get_model(cfg=model, weights=weights)

        trainer.train()
        return True

    

