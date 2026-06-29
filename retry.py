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

denoised = cv.fastNlMeansDenoising(greyscale, None, h=15, templateWindowSize=7, searchWindowSize=21)

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(20,20))
enhanced = clahe.apply(denoised)

dark_mask = cv.inRange(enhanced, 0, 20)
light_mask = cv.inRange(enhanced, 200, 255)
combined_mask = cv.bitwise_or(dark_mask, light_mask)

# Output canvas: greyscale background, coloured crater outlines drawn on top
output = cv.cvtColor(denoised, cv.COLOR_GRAY2BGR)

contours, hierarchy = cv.findContours(combined_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

kept = 0
for cnt in contours:
    area = cv.contourArea(cnt)
    perimeter = cv.arcLength(cnt, True)
    if perimeter > 100 and area > 100:
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity > 0.4:
            cv.drawContours(output, [cnt], -1, (0, 0, 255), thickness=cv.FILLED)
            kept += 1

print(f"Total raw contours: {len(contours)}")
print(f"Contours kept after circularity filter: {kept}")
cv.imshow("original", rescale(src))
cv.imshow("dark mask", rescale(dark_mask))
cv.imshow("light mask", rescale(light_mask))
cv.imshow("combined masking", rescale(combined_mask))
cv.imshow("detected craters", rescale(output))
cv.waitKey(0)
cv.destroyAllWindows()