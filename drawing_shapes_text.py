import cv2 as cv
import numpy as np

#creating blank image
blank = np.zeros((500,500,3), dtype='uint8')
cv.imshow('blank', blank)
##1. Painting the image:
#blank[:]= 0,255,0
#cv.imshow('Green', blank)
#cv.waitKey(0)
##2. Painting a part of the image:
#blank[200:300, 300:400]= 0,0,255
#cv.imshow('Partly coloured', blank)
#cv.waitKey(0)

#3. Drawing a rectangle
cv.rectangle(blank, (0,0), (250,500), (0,255,0), thickness=2) #Creating the outline of a RECTANGLE
cv.imshow('Rectangle', blank)
cv.waitKey(0)

cv.rectangle(blank, (0,0), (250,500), (0,255,0), thickness=cv.FILLED) #Creating a filled, green RECTANGLE (alternatively, thickness = -1 works too)
cv.imshow('Rectangle', blank)
cv.waitKey(0)