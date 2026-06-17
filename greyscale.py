import cv2 as cv

img= cv.imread(r"C:\Users\chitn\Downloads\grp dp.jpeg")
cv.imshow('Original', img)

#converting to greyscale
gray = cv.cvtColor(img, cv.COLOR_BGR2HLS_FULL)
cv.imshow('greyscaled', gray)
cv.waitKey(0)

#2. Blurring
blur = cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT)
cv.imshow('Blurred', blur)
cv.waitKey(0)