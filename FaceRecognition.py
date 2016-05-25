#
# FaceRecognition.py
#
# Loading frames from the webcam, identifying each face, computing the eigenvectors obtained from the faces and comparing those with previously detected ones to obtain a good view of distinct faces per frame.

# import modules
import cv2
import csv
import sys
import time
import numpy
import os


# Class VideoHandler
# Handles the given input (webcam/movie) and splits per frame and runs the detection class
class VideoHandler:
    def __init__(self, xmlPath, eigenPath, source=0, tempPath="./"):
        self.cascade = cv2.CascadeClassifier(xmlPath)
        self.recog = Recognizer(eigenPath)
        self.saver = ImageSaver(eigenPath, tempPath)
        if source != 0:                     # Check if input is movie
            source = str(source)            # Cast to str() to avoid errors
        self.capture = cv2.VideoCapture()
        self.capture.open(source)  

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.capture.release()

    def shiftFrame(self):
        try:
            self.ret, self.frame = self.capture.read()
            self.gray_frame = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)
            return True
        except:
            print "Error: "+str(sys.exc_info()[0]),str(sys.exc_info()[1])
            return False

    def createImgs(self, gray_frame, faces):
        imgs = []
        for (x,y,w,h) in faces:
            face = gray_frame[x:x+w,y:y+h]
            imgs.append((cv2.resize(face,(50,50)),(x,y,w,h))
        return imgs

    def detectFaces(self, args):
        try:
            self.faces = self.cascade.detectMultiScale(
                self.gray_frame,
                scaleFactor=args[0],
                minNeighbors=args[1],
                minSize=args[2],
                flags=cv2.cv.CV_HAAR_SCALE_IMAGE
            )
            return True
        except:
            print "Error: "+str(sys.exc_info()[0]),str(sys.exc_info()[1])
            return False

    def recognizeFaces(self):
        try:
            for img,location in self.createImgs(self.gray_frame, self.faces):
                vector = self.recog.imgToVector(img)
                if self.recog.compareVector(vector):
                    print "Bekend persoon"
                else:
                    self.saveFace(img,location)
                    print "Onbekend"
            self.saveFaces()
                
            return True
        except:
            print "Error: "+str(sys.exc_info()[0]),str(sys.exc_info())
            return False

# Class Recognizer
# Cuts out the faces and constructs eigenvectors. Saves those and compares a vector with previously found.
class Recognizer:
    eigen_vectors = []
    
    def __init__(self, eigenPath):
        self.eigenPath = eigenPath

    def imgToVector(self, img):
        return img.reshape(1,len(img)*len(img[0]))

    def loadEigenVectors(self):
        for filename in os.listdir(self.eigenPath):
            self.eigen_vectors.append(numpy.load(self.eigenPath+filename))
        return True

    def computeDistance(self, vector, base):
        return numpy.linalg.norm(numpy.dot(base,vector.reshape(-1,1)))
        
    def compareVector(self, vector):
        if len(self.eigen_vectors) == 0:
            self.loadEigenVectors()

        least = 0
        best_fit = None
        for eigen_vectors in self.eigen_vectors:
            dist = self.computeDistance(vector,eigen_vectors[3:]) #ignore biggest three due to lighting etc.
            if dist < least or least == 0:
                least = dist
                best_fit = eigen_vectors
        return least

# Class ImageSaver
# Computes the eigenvectors from an image and saves those in a .nyd file.
class ImageSaver:
    prev_detect = []
    frame_faces = []
    
    def __init__(self, eigen_path, temp_path):
        self.eigen_path = eigen_path
        self.temp_path = temp_path

    def saveFace(self, face, location):
        try:
            self.frame_faces.append((face,location))
            return True
        except:
            return False

    def saveFaces(self):
        self.prev_detect.append(self.frame_faces)
        self.frame_faces = []
        return True

    def isOldFace(self, face):
        return True

    def getVectors(self, face):
        return True

    def createEigenVectors(self, vectors):
        matrix = None
        for img_id,vector in vectors:
            try:
                matrix = numpy.vstack((matrix, vector))
            except:
                matrix = vector
        mean, eigenvectors = cv2.PCACompute(matrix, numpy.mean(matrix, axis=0).reshape(1,-1))
        print eigenvectors

    def saveToFile(self, vectors):
        return True

cascPath = "./haarcascade_frontalface_default.xml"
eigenPath = "./vecs/"
tempPath = "./temp.csv"
        
with VideoHandler(cascPath) as vid:
    if(vid.shiftFrame() and vid.detectFaces([1.10,5,(30,30)])):
        vid.recognizeFaces(eigenPath,tempPath)
        
cv2.destroyAllWindows()
