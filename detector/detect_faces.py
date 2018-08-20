"""
Based on Caffe ssd framework which used Resnet-10 as backbone
"""

import cv2
import argparse
import numpy as np


parse = argparse.ArgumentParser()
parse.add_argument('-i', '--data', required=True, help='path of input data')
parse.add_argument('-p', '--prototxt', required=True, help='path of Caffe deploy prototxt')
parse.add_argument('-m', '--model', required=True, help='path of Caffe pre-trained model')
parse.add_argument("-c", "--confidence", type=float, default=0.5,
                   help="minimum probability to filter weak detections")
args = vars(parse.parse_args())

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

image = cv2.imread(args["data"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                             (300, 300), (104.0, 177.0, 123.0))

print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > args["confidence"]:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# show the output data
cv2.imshow("Output", image)
cv2.waitKey(0)



