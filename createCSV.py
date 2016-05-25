#!usrbinenv python

import sys
import os.path
import cv2
import numpy
import json

# This is a tiny script to help you creating a CSV file from a face
# database with a similar hierarchie
#
#  philipp@mango~facerecdataat$ tree
#  .
#  -- README
#  -- s1
#     -- 1.pgm
#     -- ...
#     -- 10.pgm
#  -- s2
#     -- 1.pgm
#     -- ...
#     -- 10.pgm
#  ...
#  -- s40
#     -- 1.pgm
#     -- ...
#     -- 10.pgm
#

BASE_PATH="./faces/"
SEPARATOR=";"
vectors = []
label = 0
cascPath = "./haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cascPath)
for dirname, dirnames, filenames in os.walk(BASE_PATH):
    subject_path = os.path.join(dirname)
    for filename in os.listdir(subject_path):
        abs_path = "%s%s" % (subject_path, filename)
        if abs_path[-4:] != ".jpg":
            continue
        #im = Image.open(abs_path)
        img = cv2.imread(abs_path,cv2.CV_LOAD_IMAGE_GRAYSCALE)
        faces = cascade.detectMultiScale(
                img,
                scaleFactor=1.10,
                minNeighbors=10,
                minSize=(30,30),
                flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        for (x,y,w,h) in faces:
            face= img[x:x+w,y:y+h]
        res_face= cv2.resize(face,(50,50))
        vectors.append(res_face.reshape(50*50))
        #im.save(abs_path[:-4]+".jpg", "JPEG")

        #print "%s%s%d" % (abs_path, SEPARATOR, int(label/11))
        label = label + 1
i = 0
matrix = None
eigenvects = []

for vector in vectors:
    try:
        matrix = numpy.vstack((matrix,vector))
    except:
        matrix = vector
    if len(matrix) == 11:
        mean, eigenvectors = cv2.PCACompute(matrix, numpy.mean(matrix, axis=0).reshape(1,-1))
        eigenvects.append(eigenvectors)
        matrix = None
    i = i+1

n=0    
for vecs in eigenvects:
    numpy.save('./vecs/'+str(n)+'.npy',vecs)
    n=n+1
        
