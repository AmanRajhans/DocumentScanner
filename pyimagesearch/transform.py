# pyimagesearch/transform.py
from __future__ import annotations
import numpy as np
import cv2

def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Return the points ordered as: top-left, top-right, bottom-right, bottom-left.
    `pts` can be shape (4, 2) or (4, 1, 2) as from contour approx.
    """
    pts = np.asarray(pts, dtype="float32").reshape(4, 2)

    # Sum and diff help identify corners:
    s = pts.sum(axis=1)         # tl has smallest sum, br has largest
    d = np.diff(pts, axis=1)    # tr has smallest diff, bl has largest

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype="float32")

def four_point_transform(
    image: np.ndarray,
    pts: np.ndarray,
    dst_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Perform a perspective transform of the region defined by four points.

    Args:
        image: Source image (H x W x C).
        pts: Four corner points of the quadrilateral, in any order.
             Accepts shape (4, 2) or (4, 1, 2).
        dst_size: Optional (width, height). If not given, it's inferred
                  from the geometry of `pts`.

    Returns:
        The warped top-down view as an image.
    """
    # Order the points consistently
    (tl, tr, br, bl) = _order_points(pts)

    # Compute target width and height from side lengths
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(round(max(width_top, width_bottom)))

    height_right = np.linalg.norm(br - tr)
    height_left = np.linalg.norm(bl - tl)
    max_height = int(round(max(height_right, height_left)))

    if dst_size is not None:
        max_width, max_height = int(dst_size[0]), int(dst_size[1])

    # Destination rectangle (top-left, top-right, bottom-right, bottom-left)
    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    # Compute homography and warp
    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype="float32"), dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped
