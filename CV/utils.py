import cv2
from imutils import perspective
import numpy as np


def cv_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sort_contours(contours, method="left-to-right"):
    reverse = True if method == "right-to-left" or method == "bottom-to-top" else False
    index = 1 if method == "top-to-bottom" or method == "bottom-to-top" else 0
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    contours, bounding_boxes = zip(*sorted(zip(contours, bounding_boxes), key=lambda b: b[1][index], reverse=reverse))
    return contours, bounding_boxes


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    if width is None and height is None:
        return image
    h, w = image.shape[:2]
    if width is not None:
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        r = height / float(h)
        dim = (int(w * r), height)
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def _distance(pt1, pt2):
    return np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)


def four_point_transform(image, pts):
    rect = perspective.order_points(pts)
    (tl, tr, br, bl) = rect
    width1 = _distance(bl, br)
    width2 = _distance(tl, tr)
    width = max(int(width1), int(width2))
    height1 = _distance(tr, br)
    height2 = _distance(tl, bl)
    height = max(int(height1), int(height2))
    # 变换后对应坐标位置
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (width, height))