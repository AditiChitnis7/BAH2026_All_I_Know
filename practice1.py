#practicing with the dataset 
import cv2 as cv
import rasterio
import numpy as np

def rescale (img, scale=0.20):
    width = int(img.shape[1]*scale)
    height = int(img.shape[0]*scale)
    dimensions = (width, height)
    return cv.resize(img, dimensions, interpolation=cv.INTER_LINEAR)

img= cv.imread(r"C:\Users\chitn\Downloads\HACKATHONDATA\OHRCDATA\browse\calibrated\20260103\ch2_ohr_ncp_20260103T1005176450_b_brw_d18.png")
src =cv.normalize(img, None, 0, 255, cv.NORM_MINMAX, dtype = cv.CV_8U)
blur= cv.GaussianBlur(src, (15,15),1)
#greyscale = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
#clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(6,6))
#enhanced = clahe.apply(greyscale)
canny = cv.Canny(blur, 60, 90) 
kernel = np.ones((9,9), np.uint8)
closed = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel)
cv.imshow("original", rescale(src))
cv.imshow("blur",rescale(blur))
cv.imshow('canny edges', rescale(canny)) 
cv.waitKey(0)