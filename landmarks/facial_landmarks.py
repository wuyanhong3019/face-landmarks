# Face landmarks
# Deep learning (object detection + dlib ladnmarks)

# Usage:
# python facial_landmarks.py -s shape_predictor_68_face_landmarks.dat \
#                            -i ../data/raw/test1.jpg
#                            -p ../detector/deploy.prototxt
#                            -m ../detector/res10_300x300_ssd_iter_140000.caffemodel

import numpy as np
import cv2
from imutils import face_utils
import dlib
import argparse


parse = argparse.ArgumentParser()
parse.add_argument('-s', "--shape-predictor", required=True,
                   help="path to facial landmark predictor")
parse.add_argument('-i', "--image", required=True,
                   help="path to input image")
parse.add_argument('-p', '--prototxt', required=True, help='path of Caffe deploy prototxt')
parse.add_argument('-m', '--model', required=True, help='path of Caffe pre-trained model')
parse.add_argument('-c', "--confidence", type=float, default=0.5,
                   help="minimum probability to filter weak detections")
args = vars(parse.parse_args())

#
predictor = dlib.shape_predictor(args["shape_predictor"])

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
        box = dlib.rectangle(startX, startY, endX, endY)
        
        # detect face landmarks
        shape = predictor(gray, box)
        shape = face_utils.shape_to_np(shape)
        
#        x, y, w, h = face_utils.rect_to_bb(box)
#        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (startX - 10, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

cv2.imshow("Output", image)
cv2.waitKey(0)










