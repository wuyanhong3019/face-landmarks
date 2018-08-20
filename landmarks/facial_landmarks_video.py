# Face landmarks
# Deep learning (object detection + dlib landmarks)

# Usage:
# python facial_landmarks.py -s shape_predictor_68_face_landmarks.dat \
#                            -p ../detector/deploy.prototxt
#                            -m ../detector/res10_300x300_ssd_iter_140000.caffemodel

import numpy as np
import cv2
import time
import dlib
import argparse
import imutils

from imutils.video import VideoStream
from imutils import face_utils


parse = argparse.ArgumentParser()
parse.add_argument('-s', "--shape-predictor", required=True,
                   help="path to facial landmark predictor")
parse.add_argument('-p', '--prototxt', required=True, help='path of Caffe deploy prototxt')
parse.add_argument('-m', '--model', required=True, help='path of Caffe pre-trained model')
parse.add_argument('-c', "--confidence", type=float, default=0.5,
                   help="minimum probability to filter weak detections")
args = vars(parse.parse_args())

predictor = dlib.shape_predictor(args["shape_predictor"])

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = imutils.resize(vs.read(), width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
            box = dlib.rectangle(startX, startY, endX, endY)
        
            # detect face landmarks
            shape = predictor(gray, box)
            shape = face_utils.shape_to_np(shape)

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # show the face number
            cv2.putText(frame, "Face #{}".format(i + 1), (startX - 10, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xff == ord("q"):
        break

vs.stop()
cv2.destroyAllWindows()











