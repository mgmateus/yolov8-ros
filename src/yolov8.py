from ultralytics import YOLO


class YoloV8(YOLO):
    def __init__(self, model : str):
        """
        Initialize super class YOLO with a model
        
        Params:
        model        -- Model to initialize the class YOLO
        """
        YOLO.__init__(self, model)

    def train(self, data : str, epochs : int, imgsz : int, load_model : str = '' ) -> bool:
        """
        Initialize super class YOLO with a model
        
        Params:
        data          -- File in coco format to start the training
        epochs        -- Number of epochs to run the training 
        imgsz         -- Size of images, using nxn
        load_model    -- Model to load if you'll use transferlearning

        Return:
        True
        """
        if load_model:
            self.load(load_model)

        super().train(data=data, epochs=epochs, imgsz=imgsz)
        return True
    

