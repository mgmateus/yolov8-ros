#! /usr/bin/env python3

import rospy
import os

from yolov8 import Detection

if __name__=='__main__':
    rospy.init_node("yolov8")

    node = rospy.get_name() + "/"

    model = rospy.get_param(node + "model")
    data = rospy.get_param(node + "data")
    epochs = int(rospy.get_param(node + "epochs"))
    patience = int(rospy.get_param(node + "patience"))
    batch = int(rospy.get_param(node + "batch"))
    imgsz = int(rospy.get_param(node + "imgsz"))
    save = bool(rospy.get_param(node + "save"))
    save_period = int(rospy.get_param(node + "save_preiod"))
    cache = bool(rospy.get_param(node + "cache"))
    device = None if rospy.get_param(node + "device") == 'None' else int(rospy.get_param(node + "device"))

    train = bool(rospy.get_param(node + "train"))
    inference = bool(rospy.get_param(node + "inference"))
    load_model = rospy.get_param(node + "load_model")

    
    if train:
        Detection.train(model=model, data=data, epochs=epochs, patience=patience, batch=batch, imgsz=imgsz,save=save, save_period=save_period,\
                                cache=cache, device=device)

    


    