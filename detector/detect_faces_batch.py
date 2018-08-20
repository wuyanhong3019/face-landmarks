"""
Based on Caffe ssd framework which used Resnet-10 as backbone
"""

import cv2
import glob
import argparse
import numpy as np

parse = argparse.ArgumentParser()
parse.add_argument('-p', '--prototxt', required=True, help='path of Caffe deploy prototxt')
parse.add_argument('-m', '--model', required=True, help='path of Caffe pre-trained model')
parse.add_argument("-c", "--confidence", type=float, default=0.5,
                   help="minimum probability to filter weak detections")
args = vars(parse.parse_args())

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

files = glob.glob(r"../data/raw/*.jpg")

for (n, file) in enumerate(files):
    image = cv2.imread(file)
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    print("[INFO] computing object detections task {}...".format(n))
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

            (startX, startY, endX, endY) = box.astype("int")

            # Face roi region process
            roi = image[startY: endY, startX:endX]

            import os
            if not os.path.exists("../data/roi"):
                os.mkdir("../data/roi")

            file_path, temp_file_name = os.path.split(file)
            cv2.imwrite("../data/roi/{}.png".format(temp_file_name[:-4] + str(i + 1)), roi)








