import numpy as np
import cv2


def nms(img_hough, sigma=9):
    dilate = cv2.dilate(img_hough.astype(np.float32), np.ones((sigma, sigma)), iterations=1)
    mask = (img_hough == dilate).astype(np.uint8)
    return img_hough * mask


def hough_lines(img_hough: np.ndarray, n_lines):
    img_hough_flat = nms(img_hough).flatten()
    indices = np.argpartition(img_hough_flat, -n_lines)[-n_lines:]
    indices = indices[np.argsort(img_hough_flat[indices])][::-1]
    rhos, thetas = np.unravel_index(indices, img_hough.shape)
    return list(zip(rhos, thetas))