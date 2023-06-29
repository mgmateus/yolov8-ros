#! /usr/bin/env python3

import rospy
import os

from yolov8 import YoloV8

DIR = os.path.dirname(os.path.realpath(__file__))+"/datasets/"

if __name__=='__main__':
    rospy.init_node("yolov8")

    node = rospy.get_name() + "/"

    model = str(rospy.get_param(node + "model"))
    data = str(rospy.get_param(node + "data"))
    epochs = int(rospy.get_param(node + "epochs"))
    imgsz = int(rospy.get_param(node + "imgsz"))

    train = bool(rospy.get_param(node + "train"))
    inference = bool(rospy.get_param(node + "inference"))
    load_model = str(rospy.get_param(node + "load_model"))


    data = DIR+data
    yolov8_ros = YoloV8(model)

    if train:
        yolov8_ros.train(data, epochs, imgsz, load_model)


    