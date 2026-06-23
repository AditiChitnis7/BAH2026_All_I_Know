# Computer Vision Learning Log: PS8 Prep (BAH 2026)

Practice scripts and notes built while preparing for the **Hack2Skill x ISRO Bharatiya Antariksh Hackathon 2026**, Problem Statement 8: *Subsurface Ice Detection in the Lunar South Polar Region using Chandrayaan-2 DFSAR and OHRC data.*

This log tracks the path from basic OpenCV fundamentals to a working crater-detection pipeline on real Chandrayaan-2 OHRC imagery.

---

## Environment

- Conda environment: `compvis_env` (Python 3.9)
- Key libraries: `opencv-python`, `rasterio`, `numpy`, `scipy`, `gdal`, `requests`, `pvl` (for PDS label files)
- Install note: inside conda, plain `pip install <package>` is sufficient; no `--break-system-packages` flag needed.

---

## 1. OpenCV Fundamentals

Covered early on, applied throughout every script since:

| Concept | Function(s) | Notes |
|---|---|---|
| Grayscale conversion | `cv.cvtColor(img, cv.COLOR_BGR2GRAY)` | Required before most filtering/edge steps; many OpenCV functions (e.g. CLAHE) only accept single-channel images. |
| Drawing shapes | `cv.rectangle()`, `cv.circle()`, `cv.line()` | Used for annotating detected features (landing sites, rover paths, bounding boxes). |
| Edge detection | `cv.Canny(img, low_thresh, high_thresh)` | Sensitive to contrast; fails on low-gradient transitions (see "Hard Problem" section below). |
| Blurring | `cv.GaussianBlur(img, ksize, sigmaX)` | `ksize` must be odd x odd. `sigmaX=0` lets OpenCV auto-derive blur strength from kernel size; setting it manually gives direct control over blur intensity independent of kernel size. |
| Contour detection | `cv.findContours()`, `cv.drawContours()` | Traces boundaries from a binary/edge image. Filter with `cv.contourArea()` to drop noise-sized blobs. |
| Morphological closing | `cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)` | Bridges small gaps in broken edge lines. Different problem from noise; closing fixes *disconnected* edges, not *excess* noisy edges. |
| Contrast enhancement | `cv.createCLAHE(clipLimit, tileGridSize)` + `.apply()` | Local contrast boost; useful for low-contrast lunar shadow/light gradients, but only works on single-channel (`CV_8UC1`) images. Can over-amplify noise in flat/empty regions if `clipLimit` is too high or tiles too small. |
| Resizing for display | `cv.resize(img, dims, interpolation=cv.INTER_LINEAR)` | Used to shrink very wide satellite/lunar strips for screen viewing. |

**Debugging notes from this log:**
- `cv.GaussianBlur` kernel size must be odd (e.g. `(21,21)`, not `(20,20)`); even sizes have no defined center pixel.
- `cv.imread()` defaults to 3-channel BGR even for visually grayscale images; always confirm with `img.shape` before passing to single-channel-only functions like CLAHE.
- Don't call `cv.imread()` on an already-loaded array (e.g. `cv.imread(blur)`); `imread` is for loading from disk, not re-reading in-memory arrays.
- Avoid naming your own script the same as an imported library (`rasterio.py` causes a circular import).

---

## 2. Working with Satellite/Planetary Data (`rasterio`)

```python
import rasterio
with rasterio.open("file.tif") as src:
    print(src.width, src.height, src.count, src.crs)
    data = src.read(1)
```

- **Bands**: each band is one stacked layer of the image; RGB optical data has 3, radar data may have just 1 (per polarization), scientific instruments can have many more (thermal, NIR, etc.).
- **CRS (Coordinate Reference System)**: confirms the data is real, geolocated satellite/planetary data; e.g. `EPSG:32618` (UTM zone) for the Earth-based practice file.
- **Normalizing for OpenCV**: satellite data often exceeds the 0 to 255 range OpenCV expects.
  ```python
  normalized = cv.normalize(data, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
  ```
- **NumPy/rasterio axis order gotcha**: rasterio reports `width x height`; the resulting NumPy array is `(height, width)`; opposite order.

### Reading ISRO PDS4 data (DFSAR `.XML` + `.DAT`)
- ISRO mission data ships as a pair: a `.DAT`/`.IMG` file (raw binary pixel values) and a `.XML`/`.LBL` label file (metadata: shape, dtype, byte order).
- The `.DAT` file is *meaningless without* its label; there's no header to self-describe the binary layout.
- `rasterio.open()` pointed at the **`.xml`/`.lbl` file** (not the `.dat`) can often auto-detect the PDS4 driver and load the data directly; this is the preferred path over manual byte parsing.
- Fallback manual parsing uses `np.fromfile(path, dtype=...)` + `.reshape((height, width))`, with dtype/shape read out of the label file. PDS data is often **big-endian** (`>i2`, `>f4` etc.); verify against the label.
- Windows path strings need a raw string prefix (`r"C:\path\to\file"`) or forward slashes; backslashes are otherwise interpreted as escape sequences.

---

## 3. Dataset Notes; PS8 specific

### DFSAR (Dual Frequency SAR)
- Radar instrument; does not need sunlight, can probe subsurface composition (relevant to ice detection via polarimetric signatures).
- A raw sample file loaded for this project had shape `4885 x 509909`, dtype `int8`, no embedded georeferencing; consistent with an unprocessed orbital radar strip, not yet a finished, geocoded map product. Likely needs further processing (e.g. via ISRO's MIDAS software) before polarimetric analysis is meaningful.
- Penetration depth is not a fixed guarantee; it depends on radar wavelength and regolith properties, and should be stated explicitly rather than assumed to cover the full target depth automatically.

### OHRC (Orbiter High Resolution Camera)
- Optical instrument; high resolution, but only sees illuminated terrain; cannot see into permanently shadowed regions (PSRs).
- The **calibrated browse PNG** (e.g. `..._b_brw_d18.png`) is pre-rendered and loads directly with plain `cv.imread()`; no rasterio/GDAL parsing needed for this product.
- Other accompanying files (LBR, OAT, OATH, SPM) are spacecraft telemetry/orbit-attitude metadata; not needed for basic image-based CV work; relevant later for precise geolocation.

### Geolocation CSV (Pixel/Scan → Longitude/Latitude)
- Sparse lookup grid (sampled every 100 pixels in both directions), mapping image pixel coordinates to real lunar coordinates.
- Requires **2D interpolation** (not just 1D along one axis) since both `Pixel` and `Scan` vary:
  ```python
  from scipy.interpolate import griddata
  points = geo_table[["Scan", "Pixel"]].values
  lat = griddata(points, geo_table["Latitude"].values, (scan_val, pixel_val), method='linear')
  lon = griddata(points, geo_table["Longitude"].values, (scan_val, pixel_val), method='linear')
  ```
- For converting many points at once (e.g. an entire crater boundary), batch the query rather than looping `griddata` calls one at a time.

---

## 4. Open Problem; Low-Contrast Crater Edges

The core unsolved challenge in this log: doubly-shadowed crater rims in OHRC imagery often show a **gradual grey-to-black gradient**, not a sharp boundary; which is fundamentally a different kind of edge than what Canny is built to detect well. Attempts so far:

1. **Blur tuning** (`(21,21)` kernel); removes fine speckle noise, keeps visible crater shapes intact to the eye, but doesn't fix Canny's failure to trigger on weak gradients.
2. **CLAHE contrast boosting**; intended to sharpen the gradual fade into a stronger transition. Tuning `clipLimit` and `tileGridSize` shifted the noise pattern but didn't resolve it; at low settings Canny still missed real rims, at higher settings it amplified noise even in empty/flat sky regions.
3. **Isolating the variable**; running Canny on plain blurred grayscale (no CLAHE at all) still showed scattered salt-and-pepper-style noise, proving CLAHE was not the root cause. The noise appears to be inherent to the source image at the pixel level (sensor/compression grain), not something contrast adjustment alone can fix.

**Next direction identified, not yet implemented:** a dedicated OpenCV denoising function (rather than a generic blur) that distinguishes random pixel-level noise from real structure before edge detection; to be explored in a future session.

---

## 5. Team & Project Context

- **PS8**: Subsurface Ice Detection in the Lunar South Polar Region using Chandrayaan-2 DFSAR and OHRC data.
- **Team split**: radar/physics & volume estimation (Teammate A), CNN-based ice classification (Teammate B), software architecture & UI/demo (Teammate C), CV/terrain pipeline + data preprocessing (this repo's owner).
- **Guiding principle**: state dataset limitations explicitly (illumination history, true elevation/slope, radar penetration depth) rather than overstating certainty; flagged repeatedly as a credibility differentiator for the hackathon presentation.

---

*This README will be updated as new scripts and concepts are added.*
