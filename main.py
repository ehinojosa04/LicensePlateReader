import cv2 as cv
import numpy as np
import ImageUtils as iu

image_path = "./test_images/test2.webp"

grayscale = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
if grayscale is not None:
    bin_image = iu.umbralization(grayscale)

    fragments = iu.extractFragments(bin_image, "rect")
    fragments = sorted(fragments, key=lambda f: f.size, reverse=True)

    
    for idx, fragment in enumerate(fragments[:3]):
        if fragment.size > 500:
            bg = np.zeros(bin_image.shape, dtype=np.uint8)
            for y,x in fragment.coords:
                bg[y][x] = 255

            iu.showScaledImage(f'Fragment {idx}', bg, 1/3)
    
    iu.showScaledImage("gsc",grayscale, 1/3)
    iu.showScaledImage("bin",bin_image, 1/3)

cv.waitKey(0)
