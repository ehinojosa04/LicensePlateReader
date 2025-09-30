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
