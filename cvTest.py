# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np

# Read Image File
img = cv2.imread('IMA.jpg')
# img = cv2.imread('IMA.jpg',0)  Gray
cv2.imshow('Original',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Write Image File
img = cv2.imwrite('image.png',img)
cv2.imshow('Original',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Draw Retangle
pic =np.zeros((500,500,3),dtype='uint8')
cv2.rectangle(pic,(0,0),(500,150),(123,200,98),3,lineType=8,shift=0)
cv2.imshow('dark',pic)

cv2.waitKey(0)
cv2.destroyAllWindows()


# Draw Line
pic = np.zeros((500,500,3),dtype='uint8')
cv2.line(pic,(350,350),(500,350),(0,0,255))
cv2.imshow('dark',pic)

cv2.waitKey(0)
cv2.destoryAllWindows()


# Draw Circle
pic = np.zeros((500,500,3) , dtype='uint8')
color = (255,0,255)
cv2.circle(pic,(250,250),50,color)
cv2.imshow('dark',pic)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Writing Text
pic= np.zeros((500,500,3),dtype='uint8')
font = cv2.FONT_HERSHEY_DUPLEX
cv2.putText(pic,'Udemy',(100,100),font,3,(255,255,255),4,cv2.LINE_8)
cv2.imshow('dark',pic)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Drawing Combination
pic = np.zeros((500,500,3),dtype='uint8')
cv2.rectangle(pic,(0,0),(500,150),(123,200,98),3,lineType=8,shift=0)
font = cv2.FONT_HERSHEY_DUPLEX
cv2.putText(pic,'Udemy',(100,100),font,3,(255,255,255),3,cv2.LINE_8)
cv2.circle(pic,(250,250),50,(255,0,255))
cv2.line(pic,(133,138),(388,133),(0,0,255))
cv2.imshow('dark',pic)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Image Transformation
pic = cv2.imread('IMA.jpg')
cols = pic.shape[1]
rows = pic.shape[0]
M = np.float32([[1,0,150],[0,1,70]])
shifted = cv2.warpAffine(pic,M,(cols,rows))
cv2.imshow('shifted',shifted)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Image Rotation
pic = cv2.imread('IMA.jpg')
rows = pic.shape[1]
cols = pic.shape[0]
center = (cols/2,rows/2)
angle = 90
M =cv2.getRotationMatrix2D(center,angle,1)
rotate = cv2.warpAffine(pic,M,(cols,rows))
cv2.imshow('rotated',rotate)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Image Threshold
pic = cv2.imread('IMA.jpg',0)
threshold_value = 100
(T_value,binary_threshold) = cv2.threshold(pic,threshold_value,255,cv2.THRESH_BINARY)
cv2.imshow('binary',binary_threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Gaussian Blur
pic = cv2.imread('IMA.jpg')
matrix = (7,7)
blur = cv2.GaussianBlur(pic,matrix,0)
cv2.imshow('blurred',blur)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Median Blur
pic = cv2.imread('IMA.jpg')
kernal = 3
median = cv2.medianBlur(pic,kernal)
cv2.imshow('median',pic)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Bilateral Filtering
pic = cv2.imread('IMA.jpg')
dimpixel = 7
color = 100
space = 100
filter = cv2.bilateralFilter(pic,dimpixel,color,space)
cv2.imshow('filter',filter)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Canny
pic = cv2.imread('IMA.jpg')
thresholdval1 = 50
thresholdval2 = 100
canny = cv2.Canny(pic,thresholdval1,thresholdval2)
cv2.imshow('canny',canny)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Load a Video
cap = cv2.VideoCapture('123.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('vid',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()



# Save Video
cap = cv2.VideoCapture('123.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 30
framesize = (720,480)
out = cv2.VideoWriter('sample.avi',fourcc,fps,framesize)
while(cap.isOpened()):
    ret,frame = cap.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF== ord('q'):
        break
out.release()
cap.release()
cv2.destroyAllWindows()



# Face Detection

#face_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
pic = cv2.imread('AKD_78221.jpg')
scale_factor = 1.3

while 1:
    faces = face_cascade.detectMultiScale(pic,scale_factor,5)
    
    for(x,y,w,h) in faces:
        cv2.rectangle(pic,(x,y),(x+w,y+h),(255,0,0),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(pic,'Haru',(x,y),font,2,(255,255,255),2,cv2.LINE_AA)
    
    print("Number of faces found {}" , format(len(faces)))
    cv2.imshow('faces',pic)
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

cv2.destroyAllWindows()

# Face Detection Video
face_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')
videocapture = cv2.VideoCapture(0)
scale_factor = 1.3

while 1:
    ret,pic = videocapture.read()
    faces = face_cascade.detectMultiScale(pic,scale_factor,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(pic,(x,y),(x+w,y+h),(255,0,0),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(pic,'test0',(x,y),font,2,(255,255,255),2,cv2.LINE_AA)
    
    print("Number of faces found {}" ,format(len(faces)))
    cv2.imshow('face',pic)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()




    
















