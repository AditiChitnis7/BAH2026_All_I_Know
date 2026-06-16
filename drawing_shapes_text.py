import cv2 as cv
import numpy as np

#creating blank image
blank = np.zeros((500,500,3), dtype='uint8')
cv.imshow('blank', blank)
#1. Painting the image:
blank[:]= 0,255,0
cv.imshow('Green', blank)
cv.waitKey(0)