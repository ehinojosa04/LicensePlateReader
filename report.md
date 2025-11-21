| |
|:--:|
| ![Logo Universidad Panamericana](./Escudo_Universidad_Panamericana.png) |
| **ISGC** |
| **Técnicas de Interpretación Avanzadas** |
| **Profesor: José Abdón Ramirez Ruiz** |
| **José Salcedo Uribe** |
| **Emiliano Hinojosa Guzmán** |
| **November 21, 2025** |

# **LICENSE PLATE RECOGNITION USING IMAGE PROCESSING AND OCR**

**Project Report**
# **Table of Contents**

1. **Introduction**
2. **Theoretical Framework**
   2.1. Otsu Thresholding
   2.2. Connected Component Labeling (BFS)
   2.3. Convex Hull Computation
   2.4. Minimum-Area Rectangle (minAreaRect)
   2.5. Point Ordering Algorithm
   2.6. Perspective Warp / Homography
   2.7. Optical Character Recognition (EasyOCR)
3. **Flowchart**
4. **Pseudocode**
5. **Program**
   - Insert code screenshots here
6. **Development**
   6.1. Program Logic
   6.2. Tools and Methods Used
   6.3. What Worked
   6.4. What Didn’t Work
7. **Conclusion**
8. **References (APA Format)**

---

# **1. Introduction**

This project implements a basic license plate detection and recognition system using image processing techniques and Optical Character Recognition (OCR).
The system locates the license plate region within an input image, extracts it using geometric analysis, rectifies the plate to reduce distortion, and finally uses EasyOCR to recognize the alphanumeric characters.

The project combines classical computer vision algorithms—thresholding, connected component analysis, convex hulls, and minimum-area rectangles—with machine learning–based OCR. The pipeline is designed to operate on real-world images where plates may appear at an angle, partially occluded, or under varying lighting conditions.

---

# **2. Theoretical Framework**

This section describes all algorithms used in the system. Each algorithm is presented as a subsection.

---

## **2.1. Otsu Thresholding (Global Thresholding Method)**

Otsu’s thresholding is an algorithm used to convert a grayscale image into a binary image by automatically selecting an optimal threshold.
The method analyzes the histogram of pixel intensities and searches for the threshold value that maximizes between-class variance:

* The image is assumed to contain two classes: foreground and background.
* For each possible threshold, the algorithm computes class probabilities and class means.
* The threshold that maximizes the between-class variance is selected.

This provides a robust binarization even under non-uniform brightness conditions.

---

## **2.2. Connected Component Labeling (Breadth-First Search)**

After thresholding, the system identifies continuous white regions using connected component labeling.
A **Breadth-First Search (BFS)** is applied:

* A pixel is selected as a seed.
* All connected neighbors (4-connected or 8-connected) of the same value are explored.
* Each connected region is stored as a *fragment* consisting of its pixel coordinates.

This is used to isolate candidate regions that may contain a license plate.

---

## **2.3. Convex Hull Computation**

For each fragment, its shape is refined by computing the convex hull using OpenCV’s `cv.convexHull`.
A convex hull is the smallest convex polygon that contains all points of a region.

Convex hulls help:

* Smooth irregular shapes.
* Estimate rectangularity.
* Provide a better approximation for region geometry.

The ratio of the fragment area to the convex hull area is later used as a **rectangularity score**, helping determine whether a fragment resembles a license plate.

---

## **2.4. Minimum-Area Rectangle (minAreaRect)**

OpenCV’s `cv.minAreaRect` computes the smallest-area rotated rectangle that encloses a set of points.

From each fragment we compute:

* Center of the rectangle
* Width and height
* Rotation angle
* Four corner points (using `cv.boxPoints`)

This rectangle approximates the plate’s location and orientation even when the plate appears tilted or skewed in the image.

---

## **2.5. Point Ordering Algorithm**

The four rectangle vertices must be ordered consistently to compute a correct perspective transform.
The ordering used is:

1. Top-left
2. Top-right
3. Bottom-right
4. Bottom-left

This is achieved by comparing:

* The sum of coordinates → identifies TL and BR
* The difference of coordinates → identifies TR and BL

This ordering is crucial to avoid distortions when rectifying the plate.

---

## **2.6. Perspective Warp / Homography**

To “unskew” the plate, a **homography matrix** is estimated using the ordered quad and a target rectangle.

OpenCV’s `cv.getPerspectiveTransform` computes a 3×3 matrix **H**, and
`cv.warpPerspective` uses this matrix to project the quadrilateral region into a front-facing, axis-aligned rectangle.

This step eliminates:

* Tilt
* Perspective distortion
* Skew
* Rotational misalignment

The result is a clean, rectified plate image suitable for OCR.

---

## **2.7. Optical Character Recognition (EasyOCR)**

EasyOCR is a deep learning OCR engine based on:

* Convolutional Neural Networks (CNNs)
* LSTM-based sequence modeling
* CTC (Connectionist Temporal Classification) decoding

In this project, EasyOCR reads alphanumeric characters of the rectified plate.

Before OCR:

* The image is converted to grayscale
* Upscaled to increase text clarity
* Converted to a 3-channel image
* Restricted to `A–Z` and `0–9` allowlist

EasyOCR outputs the detected character sequence, which is post-processed to remove noise and convert it to uppercase.

---

# **3. Flowchart**

```
             +---------------------------+
             |       Load Image          |
             +-------------+-------------+
                           |
                           v
             +---------------------------+
             |      Otsu Thresholding    |
             +-------------+-------------+
                           |
                           v
             +---------------------------+
             | Connected Component Label |
             +-------------+-------------+
                           |
                           v
             +---------------------------+
             |  Compute Convex Hulls     |
             +-------------+-------------+
                           |
                           v
             +---------------------------+
             | Score Fragments (Rect + AR)|
             +-------------+--------------+
                           |
                           v
           +-------------------------------+
           | Select Best Plate Candidate   |
           +---------------+---------------+
                           |
                           v
             +---------------------------+
             | minAreaRect → Quad Points |
             +-------------+-------------+
                           |
                           v
           +-------------------------------+
           | Perspective Warp (Rectify)    |
           +---------------+---------------+
                           |
                           v
             +---------------------------+
             |     EasyOCR Recognition   |
             +-------------+-------------+
                           |
                           v
             +---------------------------+
             |        Output Text        |
             +---------------------------+
```

---

# **4. Pseudocode**

```
FUNCTION main():
    image ← load_grayscale(image_path)
    binary ← otsu_threshold(image)

    fragments ← extract_connected_components(binary)

    best_fragment ← NULL
    best_score ← +∞

    FOR each fragment IN largest 10 fragments:
        contour ← find_contour(fragment)
        convex ← convex_hull(contour)

        rectangularity ← area(fragment) / area(convex)

        quad ← minAreaRect(fragment)
        aspect ← max(width(quad), height(quad)) /
                  min(width(quad), height(quad))

        score ← |1 - rectangularity| + 0.3 * |aspect - 4|

        IF score < best_score:
            best_score ← score
            best_fragment ← fragment

    quad ← minAreaRect(best_fragment)
    rectified ← warp_quad(image, quad)

    text, ocr_input ← easyocr_read(rectified)

    PRINT text
END FUNCTION
```

---

# **5. Program (Screenshots)**

![Plate detector working](./screen0.jpeg "First test")
![Plate detector working](./screen1.jpeg "Second test")
![Plate detector working](./screen2.jpeg "Third test")

---

# **6. Development**

This project was developed incrementally, starting from image preprocessing and ending with OCR.

### **6.1. Logic of the Program**

The main logic follows these steps:

1. **Load & Preprocess Image**
   Convert the input to grayscale and binarize using Otsu thresholding.

2. **Fragment Extraction**
   BFS-based connected components identify white regions that may correspond to the plate.

3. **Geometric Filtering**
   Each fragment is compared to a rectangle by:

   * Area ratio (fragment vs convex hull)
   * Aspect ratio close to ~4:1
     The most plate-like fragment is selected.

4. **Rectangle Fitting**
   `minAreaRect` estimates the orientation and boundary of the plate.

5. **Perspective Rectification**
   The skewed quadrilateral is transformed into an upright rectangle.

6. **OCR**
   EasyOCR reads the characters. The allowlist ensures only valid characters are returned.

---

### **6.2. Tools and Methods Used**

| Component                | Why it was used                          |
| ------------------------ | ---------------------------------------- |
| Otsu Thresholding        | Automatic and robust binarization        |
| BFS Connected Components | Simple and effective region segmentation |
| Convex Hull              | Helps estimate rectangularity            |
| minAreaRect              | Detects rotated bounding boxes           |
| Homography Warp          | Removes perspective distortion           |
| EasyOCR                  | Fast and accurate text recognition       |

---

### **6.3. What Worked**

* Fragment extraction reliably isolated the plate region.
* `minAreaRect` provided good alignment even on skewed plates.
* EasyOCR achieved readable results on clean rectified images.

### **6.4. What Didn’t Work Initially**

* Perspective warping produced squashed images until quad ordering was fixed.
* Tesseract OCR performed poorly; switching to EasyOCR solved this.
* SSL certificate issues prevented model downloads on macOS; manual downloading fixed it.

---

# **7. Conclusion**

This project successfully implemented a complete license plate recognition system using classical image processing and modern OCR. It can detect plate regions, correct geometric distortions, and identify alphanumeric characters.

What we accomplished:

* Automated plate region detection
* Rectification using homography
* OCR with EasyOCR

What we would have liked to add:

* A dataset-based evaluation
* Better handling of low-light or blurred images
* Image de-skewing still has some quirks to it, yielding weird results when the angle is really acute
* Given the time restrictions on this project; getting it to be really accurate was not possible. So improving accuracy is the main focus on what I would have focused if we kept working on the project

The project provided valuable experience in integrating traditional computer vision with neural network models, debugging OpenCV transformations, and handling real-world OCR challenges.

---

# **References (APA)**

Otsu, N. (1979). *A threshold selection method from gray-level histograms*. IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62–66.

Suzuki, S., & Abe, K. (1985). *Topological structural analysis of digitized binary images by border following*. Computer Vision, Graphics, and Image Processing, 30(1), 32–46.

OpenCV Documentation. (2024). *Image processing and transformation functions*. [https://docs.opencv.org/](https://docs.opencv.org/)

JaidedAI. (2020). *EasyOCR*. GitHub Repository. [https://github.com/JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR)

Bradski, G. (2000). *The OpenCV Library*. Dr. Dobb’s Journal.
