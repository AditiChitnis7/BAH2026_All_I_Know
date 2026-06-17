import cv2 as cv

img= cv.imread(r"C:\Users\chitn\Downloads\grp dp.jpeg")
cv.imshow('Original', img)

#converting to greyscale
gray = cv.cvtColor(img, cv.COLOR_BGR2HLS_FULL)
cv.imshow('greyscaled', gray)
cv.waitKey(0)

#2. Blurring
blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
cv.imshow('Blurred', blur)
cv.waitKey(0)

#3. Edge cascade: detecting edges in the image
canny = cv.Canny(img, 125,175)
cv.imshow('Canny Edges', canny)
cv.waitKey(0)