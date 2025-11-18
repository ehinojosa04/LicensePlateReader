import cv2 as cv
import numpy as np
from numpy.typing import NDArray
from typing import Literal, Tuple, List
from collections import deque


Image = NDArray[np.uint8]
    

def umbralization(img: np.ndarray):    
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
    def __init__(self) -> None:
        self.edge: np.ndarray
        self.coords: List[Tuple[int,int]] = []
        self.size = 0


def extractFragments(image: np.ndarray, mask_type: Literal["cross", "rect"]) -> List[fragment] :
    fragments = []
    h, w = image.shape[:2]
    visited = np.zeros((h, w), dtype=bool)
    mask = [(0,0)]

    match mask_type:
        case "cross":
            mask = [(0,-1),(-1,0),(1,0),(0,1)]
        case "rect":
            mask = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]

    for y0 in range(h):
        for x0 in range(w):
            if visited[y0,x0]:
                continue
                
            if image[y0][x0] == 0:
                visited[y0,x0] = True
                continue

            visited[y0,x0] = True
            
            f = fragment()
            dq = deque([(y0,x0)])

            while dq:
                y,x = dq.popleft()
                f.coords.append((y,x))

                for dx,dy in mask:
                    ny, nx = y+dy, x+dx

                    if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx] and image[ny, nx] == 255:
                        visited[ny, nx] = True
                        dq.append((ny, nx))
            
            f.size = len(f.coords)
            fragments.append(f)
    
    return fragments


def order_points(pts):
    pts = np.asarray(pts, np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(s)],            
        pts[np.argmin(d)],            
        pts[np.argmax(s)],            
        pts[np.argmax(d)]             
    ], dtype=np.float32)

def quad_from_min_area_rect(f):
    pts = np.array([(x, y) for (y, x) in f.coords], dtype=np.float32)

    rect = cv.minAreaRect(pts)
    box  = cv.boxPoints(rect).astype(np.float32)
    quad = order_points(box)
    return quad

def warp_quad(image, quad, enforce_ratio=None):
    wA = np.linalg.norm(quad[1]-quad[0])   
    wB = np.linalg.norm(quad[2]-quad[3])   
    hA = np.linalg.norm(quad[3]-quad[0])   
    hB = np.linalg.norm(quad[2]-quad[1])   
    W = int(np.ceil(max(wA, wB)))
    H = int(np.ceil(max(hA, hB)))

    if enforce_ratio is not None:
        H = int(max(1, round(W / enforce_ratio)))

    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], np.float32)

    Hmat = cv.getPerspectiveTransform(quad, dst)
    out  = cv.warpPerspective(
        image, Hmat, (W, H),
        flags=cv.INTER_CUBIC,                     
        borderMode=cv.BORDER_REPLICATE
    )
    return out, Hmat


def showScaledImage(name: str, image: np.ndarray, scale: float | int) -> None:
    h, w = image.shape[:2]
    
    if isinstance(scale, float) and scale < 1:
        new_w = int(w * scale)
        new_h = int(h * scale)
    else:
        new_w = w // scale
        new_h = h // scale
    
    resized = cv.resize(image, (new_w, new_h))
    cv.imshow(name, resized)
