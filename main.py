import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import cv2 as cv
import numpy as np
import easyocr
import ImageUtils as iu

image_path = "./test_images/test3.jpg"

reader = easyocr.Reader(["en"], gpu=False)


def read_plate_text_easyocr(plate_img: np.ndarray) -> tuple[str, np.ndarray]:
    """
    Recognize license plate text using EasyOCR.

    The plate image is converted to grayscale, upscaled, converted to a
    3-channel image, and passed to EasyOCR with an alphanumeric allowlist.

    Args:
        plate_img: Rectified image region containing the license plate.

    Returns:
        A tuple (text, ocr_input) where:
            text: Detected alphanumeric plate string in uppercase.
            ocr_input: Image actually sent to EasyOCR for debugging.
    """
    if len(plate_img.shape) == 2:
        gray = plate_img
    else:
        gray = cv.cvtColor(plate_img, cv.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    scale = 2
    gray_big = cv.resize(gray, (w * scale, h * scale), interpolation=cv.INTER_CUBIC)
    vis_img = cv.cvtColor(gray_big, cv.COLOR_GRAY2BGR)

    results = reader.readtext(
        vis_img,
        detail=0,
        paragraph=False,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    )

    text = "".join(results)
    text = "".join(ch for ch in text if ch.isalnum()).upper()

    return text, vis_img


def main() -> None:
    """
    Detect a license plate in an image, rectify it, and read its text.

    The pipeline performs the following steps:
        1. Load and binarize the input image.
        2. Extract connected rectangular fragments.
        3. Choose the fragment that best matches a plate-like rectangle.
        4. Compute a minimum-area bounding quadrilateral and warp it.
        5. Run EasyOCR on the rectified plate region and print the result.
    """
    grayscale = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if grayscale is None:
        print(f"Could not read image at {image_path}")
        return

    bin_image = iu.umbralization(grayscale)

    fragments = iu.extractFragments(bin_image, "rect")
    fragments = sorted(fragments, key=lambda f: f.size, reverse=True)

    best_score = float("inf")
    closest_match = None

    for fragment in fragments[:10]:
        if fragment.size < 500:
            continue

        bg = np.zeros(bin_image.shape, dtype=np.uint8)
        for y, x in fragment.coords:
            bg[y, x] = 255

        contours, hierarchy = cv.findContours(
            bg,
            cv.RETR_CCOMP,
            cv.CHAIN_APPROX_SIMPLE,
        )
        if not contours or hierarchy is None:
            continue

        fragment.edge = max(contours, key=cv.contourArea)

        filled = bg.copy()
        convex_wrapping = bg.copy()

        for i, cnt in enumerate(contours):
            hull = cv.convexHull(cnt)
            target = filled if hierarchy[0][i][3] != -1 else convex_wrapping
            cv.fillConvexPoly(target, hull, 255)

        area_object = float(cv.countNonZero(filled))
        area_convex = float(cv.countNonZero(convex_wrapping))
        if area_convex == 0:
            continue

        rect_ratio = area_object / area_convex
        rect_score = abs(1.0 - rect_ratio)

        quad = iu.quad_from_min_area_rect(fragment)
        w_a = np.linalg.norm(quad[1] - quad[0])
        h_a = np.linalg.norm(quad[3] - quad[0])
        if h_a == 0:
            continue
        aspect = max(w_a, h_a) / min(w_a, h_a)

        target_aspect = 4.0
        aspect_score = abs(aspect - target_aspect)

        score = rect_score + 0.3 * aspect_score

        if score < best_score:
            best_score = score
            closest_match = fragment

    if closest_match is not None:
        mask = np.zeros_like(bin_image)
        for y, x in closest_match.coords:
            mask[y, x] = 255
        iu.showScaledImage("Best Match (mask)", mask, 3)

        quad = iu.quad_from_min_area_rect(closest_match)
        color = cv.cvtColor(grayscale, cv.COLOR_GRAY2BGR)
        quad_int = quad.astype(int)
        cv.polylines(color, [quad_int], isClosed=True, color=(0, 255, 0), thickness=2)
        iu.showScaledImage("Quad on original", color, 3)

        rectified, _ = iu.warp_quad(grayscale, quad, enforce_ratio=None)
        iu.showScaledImage("Rectified plate", rectified, 2)

        plate_text, ocr_input = read_plate_text_easyocr(rectified)
        iu.showScaledImage("Plate for OCR (EasyOCR input)", ocr_input, 2)

        print("Detected plate (EasyOCR):", plate_text)
    else:
        print("No suitable rectangular fragment (plate candidate) found.")

    iu.showScaledImage("Original grayscale", grayscale, 3)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
