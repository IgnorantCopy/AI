import cv2
import numpy as np


def hough_transform(img: np.ndarray, pho_resolution=1, theta_resolution=np.pi / 180):
    y, x = np.where(img != 0)
    points = list(zip(x, y))
    height, width = img.shape[:2]
    max_pho = np.sqrt(height ** 2 + width ** 2)
    img_hough = np.zeros((
        int(np.ceil(max_pho / pho_resolution)),
        int(np.ceil(2 * np.pi / theta_resolution))
    ), dtype=np.int32)
    rho_scale = np.arange(0, int(max_pho), pho_resolution) + 0.5 * pho_resolution
    theta_scale = np.arange(0, 2 * np.pi, theta_resolution) + 0.5 * theta_resolution
    for (px, py) in points:
        for j, theta in enumerate(theta_scale):
            rho = px * np.cos(theta) + py * np.sin(theta)
            if rho < 0 or rho > max_pho:
                continue
            i = int(rho // pho_resolution)
            img_hough[i, j] += 1

    return img_hough, rho_scale, theta_scale
