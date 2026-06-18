###NOTICEEEEEE##
#Please run this EVERYTIME YOU START AFRESH: c:\Users\chitn\OPENCV\miniconda3new\Scripts\activate && conda activate compvis_env

import cv2 as cv
import numpy as np
#
#video= cv.VideoCapture(r"C:\Users\chitn\Videos\VID_20240807_205150.mp4")
#def rescale(frame,scale=0.25):
#    width = int(frame.shape[1]* scale)
#    height = int(frame.shape[0]*scale)
#    dimensions=(width, height)
#    return cv.resize(frame, dimensions, interpolation = cv.INTER_LINEAR)
#
#while True:
#    isTrue, frame = video.read()
#    cv.imshow('Video', frame)
#    resized = rescale(frame)
#    cv.imshow('Resized Video', resized)
#    if cv.waitKey(20) & 0xFF == ord('d'):
#        break
#video.release()
#cv.destroyAllWindows()
#

capture = cv.VideoCapture(0)
while True:
    isTrue, frames = capture.read()
    
    cv.imshow('Video', frames)
    flipped_vertical = cv.flip(frames, 0)
    cv.imshow('flipped', flipped_vertical)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break
capture.release
cv.destroyAllWindows 