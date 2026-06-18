import cv2 as cv
def rescale(img, scale= 0.25):
    width= int(img.shape[1]*scale)
    height= int(img.shape[0]*scale)
    dimensions =(width, height)
    return cv.resize(img, dimensions, interpolation=cv.INTER_CUBIC)
#thresholding and contouring
img = cv.imread(r"C:\Users\chitn\Downloads\akshay.jpeg")
cv.imshow('original', rescale(img))
imgblur= cv.GaussianBlur(img, (5,5), cv.BORDER_DEFAULT)
img2gray= cv.cvtColor(imgblur, (cv.COLOR_BGR2GRAY))
ret, threshold = cv.threshold(img2gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imshow('threshold applied', rescale(threshold))

contours, heirarchy = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
cv.drawContours(img2gray, contours, -1, 255, 2)

canny= cv.Canny(img2gray, 125, 175 )
cv.imshow('contours',rescale(img2gray))
cv.imshow('canny', rescale(canny))
cv.waitKey(0)