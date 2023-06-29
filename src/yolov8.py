from ultralytics import YOLO


class YoloV8(YOLO):
    def __init__(self, model : str):
        """
        Initialize super class YOLO with a model
        
        Params:
        model        -- Model to initialize the class YOLO
        """
        YOLO.__init__(self, model)

    def train(self, data : str, epochs : int, patience: int, batch : int, imgsz : int, load_model : str = '' ) -> bool:
        """
        Initialize super class YOLO with a model
        
        Params:
        data          -- path to data file, i.e. coco128.yaml
        epochs        -- number of epochs to train for
        patience      -- epochs to wait for no observable improvement for early stopping of training
        batch         -- number of images per batch (-1 for AutoBatch)
        imgsz         -- size of input images as integer or w,h
        load_model    -- Model to load if you'll use transferlearning

        Return:
        True
        """
        if load_model:
            self.load(load_model)

        super().train(data=data, patience=patience, batch= batch, epochs=epochs, imgsz=imgsz)
        return True
    

