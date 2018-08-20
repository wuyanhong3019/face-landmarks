# Facial alignment
# Deep learning (facial detection + dlib landmarks + facial alignment)

# Usage:
# python align_faces.py
#       -s ../landmarks/shape_predictor_68_face_landmarks.dat
#       -i ../data/raw/test1.jpg
#       -p ../detector/deploy.prototxt
#       -m ../detector/res10_300x300_ssd_iter_140000.caffemodel

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

predictor = dlib.shape_predictor(args["shape_predictor"])
fa = face_utils.FaceAligner(predictor, desiredFaceWidth=256)

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

        print("[INFO] detect face landmarks and implement affine transformation...")
        faceAligned = fa.align(image, gray, box)

        # Rewrite function
        cv2.imshow("Aligned", faceAligned)
        cv2.waitKey(0)











