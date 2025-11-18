import cv2 as cv
import numpy as np
import ImageUtils as iu

image_path = "./test_images/test2.webp"
#image_path = "./test_images/test3.jpg"

grayscale = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
if grayscale is not None:
    bin_image = iu.umbralization(grayscale)

    fragments = iu.extractFragments(bin_image, "rect")
    fragments = sorted(fragments, key=lambda f: f.size, reverse=True)

    best_score = float('inf')
    closest_match = None

    for idx, fragment in enumerate(fragments[:2]):
        if fragment.size > 500:
            bg = np.zeros(bin_image.shape, dtype=np.uint8)
            for y,x in fragment.coords:
                bg[y][x] = 255

            contours, hierarchy = cv.findContours(bg, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
            fragment.edge = max(contours, key=cv.contourArea)

            edge = np.zeros_like(bin_image)
            cv.drawContours(edge, [fragment.edge], -1, 255, 1)

            filled = bg.copy()
            convex_wrapping = bg.copy()

            for i, cnt in enumerate(contours):
                hull = cv.convexHull(cnt)
                cv.fillConvexPoly(filled if hierarchy[0][i][3] != -1 else convex_wrapping, hull, 255)

            A_O = float(cv.countNonZero(filled))
            A_EC = float(cv.countNonZero(convex_wrapping))
            ratio = A_O/A_EC

            score = abs(1.0 - ratio)
            if score < best_score:
                best_score = score
                closest_match = fragment

            iu.showScaledImage(f'Fragment {idx}', bg, 1/3)
            iu.showScaledImage(f'Edge {idx}', edge, 1/2)
            iu.showScaledImage(f'Filled {idx}', filled, 1/3)
            iu.showScaledImage(f'Convex Wrapping {idx}', convex_wrapping, 1/3)
            
    if closest_match is not None:
        a = np.zeros_like(bin_image)
        for y,x in closest_match.coords:
            a[y][x] = 255
        iu.showScaledImage(f'Best Match', a, 1/3)

        quad = iu.quad_from_min_area_rect(closest_match)
        rectified, H = iu.warp_quad(grayscale, quad, enforce_ratio=None)
        iu.showScaledImage(f'Rectified', rectified, 1/3)


    
    iu.showScaledImage("gsc",grayscale, 1/3)

cv.waitKey(0)
