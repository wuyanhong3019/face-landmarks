# face landmarks
# python dlib_landmarks.py -s shape_predictor_68_face_landmarks.dat -i ../data/raw/test1.jpg

import numpy as np
import cv2
import imutils
from imutils import face_utils
import dlib
import argparse


parse = argparse.ArgumentParser()
parse.add_argument("-s", "--shape-predictor", required=True,
                   help="path to facial landmark predictor")
parse.add_argument("-i", "--image", required=True,
                   help="path to input image")
args = vars(parse.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 1)
print(type(rects))

for (i, rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

cv2.imshow('Output', image)
cv2.waitKey()










