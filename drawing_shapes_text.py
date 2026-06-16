import cv2 as cv
import numpy as np

#creating blank image
blank = np.zeros((500,500), dtype='uint8')
cv.imshow('blank', blank)
cv.waitKey(0)