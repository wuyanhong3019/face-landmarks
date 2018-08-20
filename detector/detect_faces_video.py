"""
Real-time face detector
Based on Caffe ssd framework which used Resnet-10 as backbone
"""
from imutils.video import VideoStream
import imutils

import cv2
import time
import argparse
import numpy as np

# construct the argument parse and parse the arguments
parse = argparse.ArgumentParser()
parse.add_argument('-p', '--prototxt', required=True, help='path of Caffe deploy prototxt')
parse.add_argument('-m', '--model', required=True, help='path of Caffe pre-trained model')
parse.add_argument("-c", "--confidence", type=float, default=0.8
                   ,
                   help="minimum probability to filter weak detections")
args = vars(parse.parse_args())

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# allow the camera sensor to warm up for 2 seconds
time.sleep(2.0)

while True:
    frame = imutils.resize(vs.read(), width=400)

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
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
            cv2.rectangle(frame, (startY, startX), (endY, endX),
                          (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xff == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
