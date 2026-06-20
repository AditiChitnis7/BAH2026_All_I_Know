import rasterio
import cv2 as cv
import numpy as np
import requests

url = "https://github.com/rasterio/rasterio/raw/main/tests/data/RGB.byte.tif"
response = requests.get(url, verify=True)
with open("sample.tif", "wb") as file:
    file.write(response.content)

with rasterio.open("sample.tif") as src:
    data = src.read(1)

    normalized = cv.normalize(data, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    blur = cv.GaussianBlur(normalized, (15, 15), 0)
    edges = cv.Canny(blur, 100, 200)
    contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Draw contours on a colour version so green is visible
    output = cv.cvtColor(normalized, cv.COLOR_GRAY2BGR)
    for cnt in contours:
        if cv.contourArea(cnt) > 200:
            cv.drawContours(output, [cnt], -1, (0, 255, 0), 2)

    print(f"Total contours found: {len(contours)}")

    cv.imshow('Original', normalized)
    cv.imshow('Edges', edges)
    cv.imshow('Contours', output)
    cv.waitKey(0)
    cv.destroyAllWindows()