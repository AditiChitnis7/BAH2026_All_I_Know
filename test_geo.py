import rasterio
import numpy as np
import cv2 as cv
#Downloading sample GeoTIFF data

import requests

url = "https://github.com/rasterio/rasterio/raw/main/tests/data/RGB.byte.tif"
response = requests.get(url, verify=True)
with open("sample.tif", "wb") as f:
    f.write(response.content)

#Opening and reading this data
with rasterio.open("sample.tif") as src:
    print("Shape:", src.width, "x", src.height)
    print("Number of bands:", src.count)
    print("Coordinate system:", src.crs)

    data= src.read(1) #read band 1
    print("Data type:", data.dtype)
    print("Array shape:", data.shape)

normalized = cv.normalize(data, None, 0, 255, cv.NORM_MINMAX,dtype=cv.CV_8U)
cv.imshow('GeoTIFF', normalized)
cv.waitKey(0)

cv.destroyAllWindows()