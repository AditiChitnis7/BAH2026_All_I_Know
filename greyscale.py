import cv2 as cv

img= cv.imread(r"C:\Users\chitn\Downloads\grp dp.jpeg")
cv.imshow('Original', img)

#converting to greyscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('greyscaled', gray)
cv.waitKey(0)
