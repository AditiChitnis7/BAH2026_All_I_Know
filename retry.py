#practicing with the dataset 
import cv2 as cv
import numpy as np

def rescale(img, scale=0.20):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(img, dimensions, interpolation=cv.INTER_LINEAR)

img = cv.imread(r"C:\Users\chitn\Downloads\HACKATHONDATA\OHRCDATA\browse\calibrated\20260103\ch2_ohr_ncp_20260103T1005176450_b_brw_d18.png")

src = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
greyscale = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

# h: filter strength, higher removes more noise but can blur fine detail
denoised = cv.fastNlMeansDenoising(greyscale, None, h=15, templateWindowSize=7, searchWindowSize=21)

canny = cv.Canny(denoised, 60, 120)

kernel = np.ones((5, 5), np.uint8)
closed = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel)

# Find all contours, then keep only the large ones (real craters, not rocks/grain)
contours, hierarchy = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Draw filtered contours on a colour copy so they're visible
output = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
output = cv.cvtColor(output, cv.COLOR_GRAY2BGR)

kept = 0
for cnt in contours:
    area = cv.contourArea(cnt)
    perimeter = cv.arcLength(cnt, True)
    if perimeter > 50:
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity > 0.2:
            cv.drawContours(output, [cnt], -1, (0, 255, 0), 2)
            kept += 1

colour_mask = np.zeros((denoised.shape[0], denoised.shape[1], 3), dtype=np.uint8)
dark_mask= cv.inRange(denoised, 0,40)
colour_mask[dark_mask > 0] = (255, 0, 0)
light_mask = cv.inRange(denoised, 120,255)
colour_mask[light_mask > 0] = (0,255,0)
#combined= cv.bitwise_or(dark_mask, light_mask)

print(f"Total raw contours: {len(contours)}")
print(f"Contours kept after circularity filter: {kept}")
cv.imshow("dark mask", rescale(dark_mask))
cv.imshow("light mask", rescale(light_mask))
cv.imshow("light + shadow mask", rescale(colour_mask))
#cv.imshow("original", rescale(src))
#cv.imshow("denoised", rescale(denoised))
cv.imshow("canny edges", rescale(closed))
cv.imshow("filtered craters", rescale(output))
cv.waitKey(0)
cv.destroyAllWindows()