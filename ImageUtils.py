import cv2 as cv
import numpy as np
from numpy.typing import NDArray
from typing import Literal, Tuple, List
from collections import deque

Image = NDArray[np.uint8]


def umbralization(img: Image) -> Image:
    """
    Compute a binary image using Otsu-like global thresholding.

    The function estimates an intensity threshold that maximizes the
    between-class variance of foreground and background, then returns
    a binary image where pixels above the threshold are set to 255 and
    those below are set to 0.

    Args:
        img: Grayscale input image.

    Returns:
        Binary image with values in {0, 255}.
    """
    values = img.ravel()

    L = int(values.max()) + 1

    histogram = [0] * L
    for v in values:
        histogram[v] += 1

    N = len(values)
    p = [h / N for h in histogram]

    muT = sum(i * p[i] for i in range(L))

    w0 = 0.0
    mu0 = 0.0
    best_t = 0
    best_sigma = -1.0

    for t in range(L - 1):
        w0 += p[t]
        mu0 += t * p[t]
        w1 = 1.0 - w0
        if w0 == 0.0 or w1 == 0.0:
            continue

        num = muT * w0 - mu0
        sigma = (num * num) / (w0 * w1)

        if sigma > best_sigma:
            best_sigma = sigma
            best_t = t

    return np.where(img > best_t, 255, 0).astype(np.uint8)


class fragment:
    """
    Connected-component fragment representation.

    Attributes:
        edge: Contour of the fragment as returned by OpenCV, if available.
        coords: List of pixel coordinates (row, col) belonging to the fragment.
        size: Number of pixels in the fragment.
    """

    def __init__(self) -> None:
        self.edge: np.ndarray
        self.coords: List[Tuple[int, int]] = []
        self.size: int = 0


def extractFragments(image: Image, mask_type: Literal["cross", "rect"]) -> List[fragment]:
    """
    Find connected white regions in a binary image and return them as fragments.

    Breadth-first search is used to group 8-connected or 4-connected pixels
    depending on the selected mask type.

    Args:
        image: Binary input image with values in {0, 255}.
        mask_type: Neighborhood type used for connectivity. "cross" uses
            a 4-neighborhood; "rect" uses an 8-neighborhood.

    Returns:
        List of fragment objects, one per connected component.
    """
    fragments: List[fragment] = []
    h, w = image.shape[:2]
    visited = np.zeros((h, w), dtype=bool)
    mask = [(0, 0)]

    match mask_type:
        case "cross":
            mask = [(0, -1), (-1, 0), (1, 0), (0, 1)]
        case "rect":
            mask = [
                (-1, -1), (0, -1), (1, -1),
                (-1, 0),            (1, 0),
                (-1, 1),  (0, 1),   (1, 1),
            ]

    for y0 in range(h):
        for x0 in range(w):
            if visited[y0, x0]:
                continue

            if image[y0, x0] == 0:
                visited[y0, x0] = True
                continue

            visited[y0, x0] = True

            f = fragment()
            dq = deque([(y0, x0)])

            while dq:
                y, x = dq.popleft()
                f.coords.append((y, x))

                for dx, dy in mask:
                    ny, nx = y + dy, x + dx

                    if (
                        0 <= nx < w
                        and 0 <= ny < h
                        and not visited[ny, nx]
                        and image[ny, nx] == 255
                    ):
                        visited[ny, nx] = True
                        dq.append((ny, nx))

            f.size = len(f.coords)
            fragments.append(f)

    return fragments


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order four points into a consistent rectangle vertex order.

    The returned order is:
    top-left, top-right, bottom-right, bottom-left.

    Args:
        pts: Array of shape (4, 2) with (x, y) coordinates.

    Returns:
        Array of shape (4, 2) with points ordered as TL, TR, BR, BL.
    """
    pts = np.asarray(pts, dtype=np.float32)

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def quad_from_min_area_rect(frag: fragment) -> np.ndarray:
    """
    Compute the minimum-area rotated rectangle enclosing a fragment.

    If a contour is available in `frag.edge`, it is used. Otherwise, the
    rectangle is computed from the fragment's pixel coordinates.

    Args:
        frag: Fragment instance containing either a contour or a list of
            pixel coordinates.

    Returns:
        Array of shape (4, 2) with rectangle corners ordered as
        top-left, top-right, bottom-right, bottom-left.
    """
    if hasattr(frag, "edge") and frag.edge is not None:
        pts = frag.edge.reshape(-1, 2).astype(np.float32)
    else:
        pts = np.array([(x, y) for (y, x) in frag.coords], dtype=np.float32)

    rect = cv.minAreaRect(pts)
    box = cv.boxPoints(rect)
    quad = order_points(box)

    return quad


def warp_quad(
    image: Image,
    quad: np.ndarray,
    enforce_ratio: float | None = None,
) -> tuple[Image, np.ndarray]:
    """
    Warp a quadrilateral region into an axis-aligned rectangle.

    Args:
        image: Input image.
        quad: Array of shape (4, 2) with quadrilateral points ordered
            as top-left, top-right, bottom-right, bottom-left.
        enforce_ratio: If None, the output aspect ratio is determined
            by the quadrilateral geometry. If a positive float is given,
            the output height is adjusted so that width / height is close
            to this value.

    Returns:
        A tuple (warped, H) where:
            warped: Rectified image patch.
            H: 3×3 homography used for the perspective transform.
    """
    quad = np.asarray(quad, dtype=np.float32)
    tl, tr, br, bl = quad

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(np.ceil(max(width_a, width_b)))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(np.ceil(max(height_a, height_b)))

    if max_width <= 0 or max_height <= 0:
        raise ValueError("Invalid quad dimensions for warp.")

    if enforce_ratio is not None and enforce_ratio > 0:
        max_height = max(1, int(round(max_width / enforce_ratio)))

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    H = cv.getPerspectiveTransform(quad, dst)
    warped = cv.warpPerspective(
        image,
        H,
        (max_width, max_height),
        flags=cv.INTER_CUBIC,
        borderMode=cv.BORDER_REPLICATE,
    )

    return warped.astype(np.uint8), H


def showScaledImage(name: str, image: Image, scale: float | int) -> None:
    """
    Display an image resized by a given scale factor.

    Args:
        name: Window name.
        image: Input image.
        scale: If a float < 1, the image is scaled by that factor.
            If an integer, the image is downscaled by that integer divisor.
    """
    h, w = image.shape[:2]

    if isinstance(scale, float) and scale < 1:
        new_w = int(w * scale)
        new_h = int(h * scale)
    else:
        new_w = max(1, w // int(scale))
        new_h = max(1, h // int(scale))

    resized = cv.resize(image, (new_w, new_h))
    cv.imshow(name, resized)
